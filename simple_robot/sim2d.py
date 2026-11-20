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
    # Poles / trees for extra features
    for _ in range(25):
        p = rng.uniform([-48, -24], [48, 24])
        if abs(p[1]) > 16 or abs(p[0]) > 34:
            segs += box(p[0], p[1], 0.4, 0.4)
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


class Sim2D:
    def __init__(self, params: AckermannParams, cfg: SimConfig | None = None, seed: int = 0, start_pose=None):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(seed)
        self.segs, self.path = make_world(seed)
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

    def step(self, accel: float, steer: float):
        """Advance one tick; return measurements dict."""
        c = self.cfg
        self.car.step(accel, steer, c.dt)
        self.t += c.dt
        self.k += 1
        meas = {
            "v": c.speed_scale * self.car.v + self.rng.normal(0, c.speed_std) if self.car.v > 0 else 0.0,
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
