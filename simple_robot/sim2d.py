"""A tiny standalone 2D simulator (no CARLA needed) for testing the stack.

The world is a loop road surrounded by "buildings" (line segments).  It
simulates an Ackermann vehicle, a 2D LiDAR by ray casting, a noisy GNSS
receiver (reported as latitude/longitude through the same projection CARLA
uses) and noisy wheel-speed / steering-angle sensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gnss import CarlaGeoProjection
from .geometry import transform_points
from .planning import resample
from .vehicle import AckermannParams, AckermannVehicle


def box(cx, cy, w, h):
    c = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                  [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
    return [(c[i], c[(i + 1) % 4]) for i in range(4)]


def make_world(seed: int = 0):
    """Return (segments Nx4, reference loop path Mx2)."""
    rng = np.random.default_rng(seed)
    segs = []
    # Inner block of buildings
    for cx in (-20, 0, 20):
        segs += box(cx, 0, 14, 22)
    # Outer buildings along the loop, with gaps (side streets)
    for x in np.arange(-36, 37, 12):
        segs += box(x, 30, 9, 8)
        segs += box(x, -30, 9, 8)
    for y in (-12, 0, 12):
        segs += box(-54, y, 8, 9)
        segs += box(54, y, 8, 9)
    poles = rng.uniform([-48, -24], [48, 24], size=(25, 2))
    segs = np.array([[a[0], a[1], b[0], b[1]] for a, b in segs])

    # Rounded-rectangle loop road, counter-clockwise
    hw, hh, r = 40.0, 19.0, 8.0
    pts = []
    for cx, cy, a0 in ((hw - r, -hh + r, -np.pi / 2), (hw - r, hh - r, 0.0),
                       (-hw + r, hh - r, np.pi / 2), (-hw + r, -hh + r, np.pi)):
        for a in np.linspace(a0, a0 + np.pi / 2, 12):
            pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    pts.append(pts[0])
    path = resample(np.array(pts), 0.5)
    # Poles / trees for extra LiDAR features, kept off the road
    for p in poles:
        if (abs(p[1]) > 16 or abs(p[0]) > 34) and np.min(np.linalg.norm(path - p, axis=1)) > 3.5:
            segs = np.vstack([segs, [[a[0], a[1], b[0], b[1]] for a, b in box(p[0], p[1], 0.4, 0.4)]])
    return segs, path


def raycast(origin, yaw, segs, n_beams=360, max_range=40.0):
    """Ranges for ``n_beams`` rays from ``origin`` (vectorised beams x segments)."""
    ang = yaw + np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
    d = np.column_stack([np.cos(ang), np.sin(ang)])            # B x 2
    p, q = segs[:, :2], segs[:, 2:]                             # S x 2
    e = q - p
    w = p - np.asarray(origin)
    denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]   # B x S
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[None, :, 0] * e[None, :, 1] - w[None, :, 1] * e[None, :, 0]) / denom
        u = (w[None, :, 0] * d[:, None, 1] - w[None, :, 1] * d[:, None, 0]) / denom
    hit = (np.abs(denom) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
    t = np.where(hit, t, np.inf)
    r = t.min(axis=1)
    return ang - yaw, np.minimum(r, np.inf)


def sample_segments(segs: np.ndarray, step: float = 0.1) -> np.ndarray:
    """Dense points along line segments (used for ground-truth clearance)."""
    out = []
    for x0, y0, x1, y1 in segs:
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step) + 1)
        t = np.linspace(0, 1, n)[:, None]
        out.append(np.column_stack([x0 + t[:, 0] * (x1 - x0), y0 + t[:, 0] * (y1 - y0)]))
    return np.vstack(out) if out else np.zeros((0, 2))


def footprint_clearance(pose, params, points: np.ndarray) -> float:
    """Exact distance from the vehicle rectangle to the nearest point (<= 0 means collision)."""
    if len(points) == 0:
        return np.inf
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = points - pose[:2]
    bx, by = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    cx = params.length / 2.0 - params.rear_overhang
    ex = np.maximum(np.abs(bx - cx) - params.length / 2.0, 0.0)
    ey = np.maximum(np.abs(by) - params.width / 2.0, 0.0)
    inside = (ex == 0) & (ey == 0)
    dist = np.hypot(ex, ey)
    return float(-1.0 if inside.any() else dist.min())


# Obstacles on/near the loop road of make_world (the loop runs counter-clockwise).
DEFAULT_OBSTACLES = [
    (40.9, 2.0, 2.0, 4.6),     # parked car half-blocking the east side
    (-12.0, 18.4, 1.2, 1.2),   # crate on the north side
    (-40.9, 9.0, 1.5, 1.5),    # slalom on the west side ...
    (-39.1, -9.0, 1.5, 1.5),   # ... second gate
]
# Pedestrian crossing the south road: (x, y_min, y_max, speed, size)
DEFAULT_MOVERS = [(0.0, -25.0, -13.0, 1.2, 0.6)]


@dataclass
class SimConfig:
    dt: float = 0.05
    gnss_rate: float = 5.0          # Hz
    gnss_std: float = 0.5           # m
    lidar_rate: float = 10.0        # Hz
    lidar_range: float = 40.0
    lidar_std: float = 0.02
    lidar_beams: int = 360
    speed_std: float = 0.05
    speed_scale: float = 1.03       # wheel-speed scale error (tyre radius / slip)
    gnss_outage: tuple | None = None  # (t_start, t_end) with no GNSS, e.g. a tunnel
    steer_std: float = np.deg2rad(0.5)
    geo_origin: tuple = (52.0, 4.37)  # lat, lon of local origin
    lever_arm: tuple = (1.0, 0.0)     # GNSS antenna on the body frame
    obstacles: tuple = ()             # static boxes (cx, cy, w, h)
    movers: tuple = ()                # pedestrians (x, y_min, y_max, speed, size) walking back and forth in y
    start_xy: tuple = (-24.0, -19.0)  # the route starts at the path point closest to this (on a straight)


class Sim2D:
    def __init__(self, params: AckermannParams, cfg: SimConfig | None = None, seed: int = 0, start_pose=None):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(seed)
        self.static_segs, path = make_world(seed)
        i0 = int(np.argmin(np.linalg.norm(path - np.asarray(self.cfg.start_xy), axis=1)))
        self.path = np.roll(path, -i0, axis=0)
        obs = [seg for b in self.cfg.obstacles for seg in box(*b)]
        if obs:
            self.static_segs = np.vstack([self.static_segs, [[a[0], a[1], b[0], b[1]] for a, b in obs]])
        self._static_pts = sample_segments(self.static_segs)
        self.segs = self.static_segs
        self.min_clearance = np.inf
        self.collisions = 0
        self.collisions_stationary = 0
        self._in_collision = False
        if start_pose is None:
            d = self.path[1] - self.path[0]
            start_pose = (*self.path[0], np.arctan2(d[1], d[0]))
        self.car = AckermannVehicle(params, start_pose)
        self.geo = CarlaGeoProjection(*self.cfg.geo_origin)
        self.t = 0.0
        self.k = 0

    def _every(self, rate: float) -> bool:
        n = max(1, int(round(1.0 / (rate * self.cfg.dt))))
        return self.k % n == 0

    def mover_positions(self, t: float) -> np.ndarray:
        out = []
        for x, y0, y1, v, _ in self.cfg.movers:
            span = y1 - y0
            ph = (v * t) % (2 * span)
            out.append((x, y0 + (ph if ph < span else 2 * span - ph)))
        return np.array(out).reshape(-1, 2)

    def _update_movers(self) -> np.ndarray:
        if not self.cfg.movers:
            return np.zeros((0, 2))
        segs, pts = [], []
        for (x, y), m in zip(self.mover_positions(self.t), self.cfg.movers):
            b = box(x, y, m[4], m[4])
            segs += [[a[0], a[1], q[0], q[1]] for a, q in b]
        segs = np.array(segs)
        self.segs = np.vstack([self.static_segs, segs])
        return sample_segments(segs)

    def step(self, accel: float, steer: float, reverse: bool = False):
        """Advance one tick; return measurements dict."""
        c = self.cfg
        self.car.step(accel, steer, c.dt, reverse)
        self.t += c.dt
        self.k += 1
        mover_pts = self._update_movers()
        clear = min(footprint_clearance(self.car.pose, self.car.p, self._static_pts),
                    footprint_clearance(self.car.pose, self.car.p, mover_pts))
        self.min_clearance = min(self.min_clearance, clear)
        if clear <= 0 and not self._in_collision:
            self.collisions += 1
            if abs(self.car.v) < 0.1:
                self.collisions_stationary += 1   # robot standing still, something ran into it
        self._in_collision = clear <= 0
        meas = {
            "v": c.speed_scale * self.car.v + self.rng.normal(0, c.speed_std) if abs(self.car.v) > 1e-3 else 0.0,
            "steer": self.car.delta + self.rng.normal(0, c.steer_std),
            "gnss": None,
            "scan": None,
        }
        in_outage = c.gnss_outage is not None and c.gnss_outage[0] <= self.t < c.gnss_outage[1]
        if self._every(c.gnss_rate) and not in_outage:
            ant = transform_points(np.array([c.lever_arm]), self.car.pose)[0]
            e, n = ant + self.rng.normal(0, c.gnss_std, 2)
            meas["gnss"] = self.geo.from_enu(e, n)  # (lat, lon, alt) like a real receiver
        if self._every(c.lidar_rate):
            ang, r = raycast(self.car.pose[:2], self.car.pose[2], self.segs, c.lidar_beams, c.lidar_range)
            m = np.isfinite(r) & (r < c.lidar_range)
            r = r[m] + self.rng.normal(0, c.lidar_std, m.sum())
            meas["scan"] = np.column_stack([r * np.cos(ang[m]), r * np.sin(ang[m])])
        return meas
