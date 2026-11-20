"""Speed- and inertia-aware local obstacle avoidance.

Every planning cycle (each new LiDAR scan) the planner

1. builds a fan of **candidate paths**: the reference route shifted sideways
   by a set of lateral offsets.  The transition from the robot's current
   offset to the candidate's offset is a smooth-step whose length grows with
   speed so that the swerve never needs more than ``a_lat_max`` lateral
   acceleration (``L >= v * sqrt(6 |dd| / a_lat_max)``).  A committed swerve is
   anchored to the route's arc length so re-planning continues it;
2. **rolls out** each candidate: the closed-loop vehicle (pure pursuit + the
   steering-rate limit + the bicycle model) is simulated along it, so the
   trajectory is the one the robot can actually drive, not the ideal curve;
3. **collision-checks** the swept footprint (a chain of circles covering the
   vehicle) against the LiDAR obstacle points with a safety margin that grows
   with speed, giving the free distance ``s_free`` along each rollout.  Moving
   obstacles (see :class:`ObstacleTracker`) are checked where they will be
   when the robot gets there, and the robot never plans to stop in their path;
4. assigns each candidate the **highest safe speed**: the speed from which the
   robot can still stop before ``s_free`` given its reaction time (sensing,
   compute, drivetrain lag) and braking deceleration,
   ``v * t_r + v^2 / (2 a_brake) <= s_free - stop_buffer``, further limited by
   the lateral-acceleration limit on the rollout's curvature and by the route's
   own speed profile.  Candidates are validated at the speed reachable before
   the next plans and the robot is never commanded faster than that;
5. picks the cheapest **eligible** candidate - one the robot can stop on before
   its first conflict braking as hard as it can from its current speed.  The
   cost adds offset from the route, switching, unused clearance, lost speed and
   being obstructed within the horizon (the last term makes the robot swerve
   while it still has room instead of slowing down in front of an obstacle until
   it is too close to steer around it).  With no eligible candidate the robot
   brakes along the one with the most room before static obstacles, keeping its
   committed swerve on ties.

Between plans, :meth:`LocalPlanner.brake_limit` keeps checking the remaining
free distance so the longitudinal command always brakes hard enough to stop in
time.  Recovery when stuck (an Ackermann vehicle cannot turn on the spot) is
in :meth:`LocalPlanner.choose_recovery`; the stop-or-go decision for late
detected moving obstacles is in :meth:`LocalPlanner.plan`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .geometry import compose, ominus, transform_points
from .planning import Path
from .vehicle import AckermannParams, bicycle_step


@dataclass
class AvoidanceConfig:
    offsets: tuple = (0.0, -0.75, 0.75, -1.5, 1.5, -2.25, 2.25, -3.0, 3.0)  # [m], + = left
    ds: float = 0.5                  # rollout / candidate resolution [m]
    horizon_min: float = 15.0        # [m]
    margin_static: float = 0.4       # lateral clearance kept at standstill [m]
    margin_per_speed: float = 0.08   # extra clearance per m/s of speed [s]
    reaction_time: float = 0.3       # sensing + compute + actuation delay [s] (drivetrain lag is added)
    brake_decel: float = 3.0         # deceleration the plan assumes [m/s^2] (< vehicle max: reserve)
    stop_buffer: float = 1.0         # gap kept to an obstacle when stopped [m]
    a_lat_max: float = 2.0           # lateral acceleration limit [m/s^2]
    min_transition: float = 4.0      # shortest swerve [m]
    lookahead_min: float = 3.0       # pure pursuit, same law as the tracking controller
    lookahead_gain: float = 0.8
    w_offset: float = 1.0            # cost weights
    w_switch: float = 0.4
    w_clear: float = 6.0
    w_speed: float = 4.0
    w_block: float = 15.0            # obstructed within the horizon (would have to stop): swerve early
    validate_ahead: float = 0.5      # candidates are checked at the speed reachable this far ahead [s]
    profile_tolerance: float = 0.6   # drop the committed manoeuvre if the robot is this far off it [m]
    dyn_sigma_v: float = 0.3         # velocity uncertainty of moving obstacles: margin grows by this * t [m/s]
    dyn_t_max: float = 4.0           # predict moving obstacles at most this far [s]
    creep_speed: float = 0.8         # speed allowed while inside the margin but moving away [m/s]
    hard_gap: float = 0.2            # never plan closer than this, even when creeping [m]
    creep_tolerance: float = 0.03    # while creeping, the gap may shrink this much per plan (body swing) [m]
    recover_after: float = 4.0       # no progress (< stuck_distance) for this long -> back up [s]
    stuck_distance: float = 0.3      # [m]
    recover_distance: float = 4.0    # how far to back up at most [m]
    recover_speed: float = 0.8       # [m/s]
    tabu_plans: int = 100            # after getting stuck, avoid that offset for this many plans
    w_tabu: float = 25.0
    n_circles: int = 6               # circles covering the footprint


@dataclass
class Candidate:
    offset: float
    path_xy: np.ndarray
    rollout: np.ndarray        # K x 3 poses
    s: np.ndarray              # arc length of each rollout pose
    s_free: float
    clearance: float
    v_safe: float
    cost: float = math.inf
    valid: bool = True
    profile: tuple | None = None
    s_static: float = math.inf  # free distance counting static obstacles only


@dataclass
class Plan:
    path: Path
    offset: float
    v_target: float
    s_free: float
    clearance: float
    blocked: bool
    candidates: list = field(default_factory=list)
    cost: float = math.inf


class ObstacleSet:
    """Obstacles in the planning frame: static points plus moving tracks (points + velocity)."""

    def __init__(self, points: np.ndarray | None = None, dynamic: list | None = None):
        pts = np.zeros((0, 2)) if points is None else np.asarray(points, dtype=float).reshape(-1, 2)
        self.points = pts
        self.tree = cKDTree(pts) if len(pts) else None
        # list of (cKDTree of the track's points, Mx2 points, velocity (2,))
        self.dynamic = [(cKDTree(p), p, np.asarray(v, dtype=float)) for p, v in (dynamic or [])]

    def distance(self, centres: np.ndarray, upper: float = 50.0) -> np.ndarray:
        if self.tree is None:
            return np.full(len(centres), np.inf)
        d, _ = self.tree.query(centres, distance_upper_bound=upper)
        return d

    def dynamic_conflicts(self, centres: np.ndarray, t_arrive: np.ndarray, radius: float, sigma_v: float,
                          t_max: float, dt: float = 0.25, hold: float = 2.0, window=None):
        """Conflicts with moving obstacles for footprint circles ``centres`` (K x n x 2) reached at times
        ``t_arrive`` (K,).  Each obstacle is swept along its predicted path for ``t_max`` seconds, its
        radius growing by ``sigma_v * tau`` (velocity uncertainty).

        Returns ``(in_window, in_future)``: pose k is hit around the time the robot gets there, or within
        ``hold`` seconds after it gets there (i.e. it would be hit if the robot ended up *stopping* there).
        """
        k, n, _ = centres.shape
        taus = np.arange(0.0, t_max + 1e-9, dt)
        if window is None:
            window = 0.5 + 0.2 * t_arrive                 # timing uncertainty of the robot's own arrival
        window = np.broadcast_to(np.asarray(window, dtype=float), t_arrive.shape)
        hold = np.full_like(t_arrive, hold)
        in_win = np.zeros(k, dtype=bool)
        in_fut = np.zeros(k, dtype=bool)
        for tree, _, vel in self.dynamic:
            q = centres[:, :, None, :] - vel[None, None, None, :] * taus[None, None, :, None]   # K x n x J x 2
            d, _ = tree.query(q.reshape(-1, 2), distance_upper_bound=radius + sigma_v * t_max + 1.0)
            occ = d.reshape(k, n, len(taus)).min(axis=1) < radius + sigma_v * taus[None, :]   # K x J
            near = np.abs(taus[None, :] - t_arrive[:, None]) <= window[:, None]
            later = (taus[None, :] >= (t_arrive - window)[:, None]) & (taus[None, :] <= (t_arrive + hold)[:, None])
            in_win |= (occ & near).any(axis=1)
            in_fut |= (occ & later).any(axis=1)
        return in_win, in_fut


@dataclass
class TrackerConfig:
    cluster_radius: float = 0.6      # points closer than this belong to one object [m]
    max_extent: float = 3.0          # larger clusters (walls, buildings) are treated as static [m]
    gate: float = 1.2                # association gate for track centroids [m]
    alpha: float = 0.35              # velocity smoothing
    min_hits: int = 3
    min_speed: float = 0.5           # slower tracks are treated as static [m/s]
    max_age: float = 0.5             # drop tracks not seen for this long [s]
    evidence_dt: float = 0.5         # compare with the scan this long ago to confirm motion [s]
    evidence_dist: float = 0.3       # a point "moved" if no point of that scan is this close [m]
    evidence_frac: float = 0.5       # fraction of a cluster's points that must have moved
    evidence_needed: int = 3         # consecutive-ish confirmations before a track counts as moving
    static_after: float = 1.5        # a moving track that is slow for this long becomes static again [s]
    lost_memory: float = 2.0         # a lost moving track is re-identified if a cluster appears where it should be [s]


class ObstacleTracker:
    """Clusters obstacle points and tracks small clusters to estimate their velocity.

    Deliberately simple (nearest-neighbour association, smoothed finite differences):
    enough to anticipate a pedestrian or cyclist crossing the robot's path.
    """

    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.tracks: list[dict] = []
        self._next_id = 0
        self._history: list = []      # (t, cKDTree of that scan's points)
        self._lost: list[dict] = []   # recently lost moving tracks (occlusion, merged with a wall, ...)

    def _clusters(self, pts: np.ndarray) -> list[np.ndarray]:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        if len(pts) == 0:
            return []
        pairs = cKDTree(pts).query_pairs(self.cfg.cluster_radius, output_type="ndarray")
        n = len(pts)
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
        _, labels = connected_components(g, directed=False)
        return [np.flatnonzero(labels == lab) for lab in np.unique(labels)]

    def update(self, points_world: np.ndarray, t: float) -> ObstacleSet:
        cfg = self.cfg
        clusters = self._clusters(points_world)
        small = []
        static_idx = []
        for idx in clusters:
            p = points_world[idx]
            if len(idx) >= 2 and np.linalg.norm(p.max(0) - p.min(0)) <= cfg.max_extent:
                small.append((idx, p.mean(0)))
            else:
                static_idx.append(idx)
        # predict existing tracks and associate greedily by distance
        for tr in self.tracks:
            tr["pred"] = tr["c"] + tr["v"] * (t - tr["t"])
        used = set()
        assigned = {}
        cand = sorted(((np.linalg.norm(c - tr["pred"]), i, j) for j, (_, c) in enumerate(small)
                       for i, tr in enumerate(self.tracks)), key=lambda x: x[0])
        for d, i, j in cand:
            if d > cfg.gate or i in assigned or j in used:
                continue
            assigned[i] = j
            used.add(j)
        for i, j in assigned.items():
            tr = self.tracks[i]
            c = small[j][1]
            dt = max(t - tr["t"], 1e-3)
            v_meas = (c - tr["c"]) / dt
            tr["v"] = v_meas if tr["hits"] == 1 else (1 - cfg.alpha) * tr["v"] + cfg.alpha * v_meas
            tr.update(c=c, t=t, hits=tr["hits"] + 1, idx=small[j][0])
        for j, (idx, c) in enumerate(small):
            if j in used:
                continue
            tr = dict(id=self._next_id, c=c, v=np.zeros(2), t=t, hits=1, idx=idx)
            self._next_id += 1
            # re-identify a moving object that was briefly lost: inherit its motion state
            for k, lost in enumerate(self._lost):
                if np.linalg.norm(lost["c"] + lost["v"] * (t - lost["t"]) - c) < 2.0 * cfg.gate:
                    tr.update(v=lost["v"], hits=cfg.min_hits, moving=True, evidence=cfg.evidence_needed)
                    self._lost.pop(k)
                    break
            self.tracks.append(tr)
        alive = [tr for tr in self.tracks if t - tr["t"] <= cfg.max_age]
        self._lost += [tr for tr in self.tracks if t - tr["t"] > cfg.max_age and tr.get("moving")]
        self._lost = [tr for tr in self._lost if t - tr["t"] <= cfg.lost_memory]
        self.tracks = alive

        # Everything seen is an obstacle where it is *now*; tracked velocity only adds predicted future
        # positions.  (Never drop points because a track looks like it moves: viewpoint changes give
        # static objects spurious velocities, and losing a static obstacle would be unsafe.)
        # A track is "moving" once it has a consistent velocity AND most of its points are not where
        # anything was ~0.5 s ago (static surfaces seen from a new angle fail this test).  Once moving it
        # stays dynamic while tracked, so a pedestrian that pauses or turns around is still predicted.
        old_tree = self._scan_from(t - cfg.evidence_dt)
        dynamic = []
        for tr in self.tracks:
            if tr["t"] != t or tr["hits"] < cfg.min_hits:
                continue
            fast = np.linalg.norm(tr["v"]) >= cfg.min_speed
            if fast and old_tree is not None:
                d, _ = old_tree.query(points_world[tr["idx"]], distance_upper_bound=cfg.evidence_dist)
                moved = np.mean(~np.isfinite(d)) >= cfg.evidence_frac
                tr["evidence"] = tr.get("evidence", 0) + 1 if moved else max(0, tr.get("evidence", 0) - 1)
            if tr.get("evidence", 0) >= cfg.evidence_needed:
                tr["moving"] = True
            tr["slow_since"] = None if fast else (tr.get("slow_since") or t)
            if tr.get("moving") and tr["slow_since"] is not None and t - tr["slow_since"] > cfg.static_after:
                tr["moving"], tr["evidence"] = False, 0      # stopped for a while: static again
            if tr.get("moving"):
                dynamic.append((points_world[tr["idx"]], tr["v"].copy()))
        if len(points_world):
            self._history.append((t, cKDTree(points_world)))
        self._history = [h for h in self._history if t - h[0] <= 2.0 * cfg.evidence_dt]
        return ObstacleSet(points_world, dynamic)

    def _scan_from(self, t_query: float):
        """KD-tree of the stored scan closest to (but not after) ``t_query``."""
        older = [h for h in self._history if h[0] <= t_query + 1e-6]
        return older[-1][1] if older else None


class LocalPlanner:
    def __init__(self, params: AckermannParams, cfg: AvoidanceConfig | None = None, drivable=None):
        """``drivable``: optional callable mapping Nx2 points -> bool array (e.g. CARLA lane lookup)."""
        self.p = params
        self.cfg = cfg or AvoidanceConfig()
        self.drivable = drivable
        n = self.cfg.n_circles
        seg = params.length / n
        self.circle_x = -params.rear_overhang + seg * (np.arange(n) + 0.5)
        self.circle_r = math.hypot(seg / 2.0, params.width / 2.0)
        self.prev_offset = 0.0
        self.profile: tuple | None = None   # (anchor s, d_from, d_to, length) of the committed manoeuvre
        self.tabu: dict = {}                 # offset -> remaining plans it is penalised for (dead ends)
        self.go_speed: float | None = None   # committed "go through" speed in a stop-or-go dilemma
        self.last: Plan | None = None

    # ------------------------------------------------------------ physics
    @property
    def reaction_time(self) -> float:
        return self.cfg.reaction_time + self.p.accel_tau

    def margin(self, v: float) -> float:
        return self.cfg.margin_static + self.cfg.margin_per_speed * abs(v)

    def stopping_distance(self, v: float, decel: float | None = None) -> float:
        a = decel or self.cfg.brake_decel
        return v * self.reaction_time + v * v / (2.0 * a)

    def v_from_distance(self, dist: float, decel: float | None = None) -> float:
        """Largest speed whose stopping distance is ``dist`` (inverse of :meth:`stopping_distance`)."""
        if dist <= 0.0:
            return 0.0
        a, tr = decel or self.cfg.brake_decel, self.reaction_time
        return a * (-tr + math.sqrt(tr * tr + 2.0 * dist / a))

    def horizon(self, v_max: float) -> float:
        return max(self.cfg.horizon_min, self.stopping_distance(v_max) + self.cfg.stop_buffer + self.p.length)

    # ------------------------------------------------------- collision check
    def footprint_centres(self, poses: np.ndarray) -> np.ndarray:
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        cx = poses[:, 0, None] + c[:, None] * self.circle_x[None]
        cy = poses[:, 1, None] + s[:, None] * self.circle_x[None]
        return np.stack([cx, cy], axis=-1)  # K x n x 2

    def free_distance(self, poses: np.ndarray, s: np.ndarray, obstacles: ObstacleSet, v: float,
                      v_now: float | None = None):
        """(s_free, clearance, creeping) along a pose sequence.

        Poses closer than the speed-dependent margin to an obstacle are collisions, except that a
        robot which already is inside its margin (e.g. it stopped close to something) may creep
        along poses that do not get any closer than it is now - otherwise it could never leave.
        Moving obstacles are checked where they will be when the robot reaches each pose.
        """
        k, n = len(poses), self.cfg.n_circles
        centres = self.footprint_centres(poses)
        d = obstacles.distance(centres.reshape(-1, 2)).reshape(k, n)
        gap = d.min(axis=1) - self.circle_r            # distance from footprint to nearest obstacle
        margin = self.margin(v)
        creeping = bool(gap[0] < margin)
        hit = gap < margin
        if creeping:  # already too close: only moves that never get any closer are allowed
            tol = self.cfg.creep_tolerance
            hit = (gap < min(self.cfg.hard_gap, gap[0] - 1e-3)) | ((gap < margin) & (gap < gap[0] - tol))
        s_static = self._first_free(hit, s)        # free distance w.r.t. what is certainly there
        hit_static = hit.copy()
        self.last_go_clear = False
        if obstacles.dynamic:
            # Pass in front of / behind a moving obstacle only if the timing clearly works out;
            # otherwise stop short of its predicted path (not just of where it will be) so the robot
            # never ends up standing in a pedestrian's way.
            t = s / max(0.5 * (v + max(v, 1.0)), 0.5)
            in_win, in_fut = obstacles.dynamic_conflicts(centres, t, self.circle_r + margin,
                                                         self.cfg.dyn_sigma_v, self.cfg.dyn_t_max)
            if in_fut[0]:
                # Already standing in its way: stopping here is the worst option.  Only a real timing
                # conflict may stop the robot until it is out of the walker's path.
                out = int(np.argmin(in_fut)) if not in_fut.all() else len(in_fut)
                dyn_hit = in_win.copy()
                if in_win.any():
                    dyn_hit[out:] |= in_fut[out:]
                hit |= dyn_hit
            elif in_win.any():
                hit |= in_fut
            if v_now is not None and v_now > 1.0 and in_win.any():
                # "Go" option for the stop-or-go dilemma: holding the current speed, is the robot clear
                # of every moving obstacle (tight timing) and of static ones over the whole horizon?
                go_win, _ = obstacles.dynamic_conflicts(centres, s / v_now, self.circle_r + margin,
                                                        self.cfg.dyn_sigma_v, self.cfg.dyn_t_max, window=0.3)
                go_hit = hit_static | go_win
                go_hit[0] = False
                self.last_go_clear = not go_hit.any()
        hit[0] = False
        s_free = self._first_free(hit, s)
        first = int(np.argmax(hit)) if hit.any() else len(s)
        return s_free, float(gap[: max(first, 1)].min()), creeping, s_static

    @staticmethod
    def _first_free(hit: np.ndarray, s: np.ndarray) -> float:
        """Arc length of the last pose before the first hit (pose 0 never counts)."""
        h = hit.copy()
        h[0] = False
        if not h.any():
            return float(s[-1])
        first = int(np.argmax(h))
        return float(s[first - 1])

    # --------------------------------------------------------------- rollout
    def rollout(self, pose, delta: float, path_xy: np.ndarray, v: float, length: float):
        """Simulate pure pursuit + steering-rate limit + bicycle model along ``path_xy``."""
        cfg, L = self.cfg, self.p.wheelbase
        ds = cfg.ds
        x, y, yaw = (float(q) for q in pose)
        ld = cfg.lookahead_min + cfg.lookahead_gain * v
        ld_steps = max(1, int(math.ceil(ld / ds)))
        max_dd = self.p.max_steer_rate * ds / max(v, 0.5)   # steering-rate limit per metre travelled
        ms = self.p.max_steer
        n = len(path_xy)
        px, py = path_xy[:, 0].tolist(), path_xy[:, 1].tolist()
        idx = 0
        steps = int(length / ds)
        out = np.empty((steps + 1, 3))
        out[0] = (x, y, yaw)
        curv = np.zeros(steps + 1)
        for k in range(1, steps + 1):
            while idx + 1 < n and (px[idx + 1] - x) ** 2 + (py[idx + 1] - y) ** 2 <= (px[idx] - x) ** 2 + (py[idx] - y) ** 2:
                idx += 1
            j = min(idx + ld_steps, n - 1)
            dx, dy = px[j] - x, py[j] - y
            alpha = math.atan2(dy, dx) - yaw
            dc = math.atan2(2.0 * L * math.sin(alpha), max(math.hypot(dx, dy), 1e-3))
            dc = min(max(dc, -ms), ms)
            delta += min(max(dc - delta, -max_dd), max_dd)
            w = math.tan(delta) / L
            if abs(w * ds) < 1e-9:
                x += ds * math.cos(yaw)
                y += ds * math.sin(yaw)
            else:
                x += (math.sin(yaw + w * ds) - math.sin(yaw)) / w
                y -= (math.cos(yaw + w * ds) - math.cos(yaw)) / w
            yaw += w * ds
            out[k] = (x, y, yaw)
            curv[k] = w
        return out, np.arange(steps + 1) * ds, curv

    # ------------------------------------------------------------------ plan
    def _route_rel(self, ref: Path, s_abs: np.ndarray, s0: float) -> np.ndarray:
        """Signed arc length from ``s0`` along the route (wrapping on closed routes)."""
        d = np.asarray(s_abs) - s0
        if ref.closed:
            d = (d + ref.length / 2.0) % ref.length - ref.length / 2.0
        return d

    def plan(self, pose, v: float, delta: float, ref: Path, ref_idx: int, v_ref: float,
             obstacles: ObstacleSet, commit: bool = True, frame=None) -> Plan:
        """Evaluate the candidate fan and pick one.

        ``pose``, ``obstacles`` and the returned path live in the *planning frame*.  ``frame`` is the
        pose of the route's (map) frame in the planning frame; pass it when planning in a smooth
        odometry frame so GNSS corrections do not jolt the local plan (``None``: same frame).
        ``commit=False`` evaluates without changing the planner's state (used to look ahead when
        choosing a recovery manoeuvre)."""
        cfg = self.cfg
        # Validate candidates at the speed the robot can reach before the next plans; the
        # commanded speed never exceeds it, so every speed actually driven has been checked
        # (margins, swerve length and lookahead all grow with speed).
        v_plan = float(np.clip(abs(v) + self.p.max_accel * cfg.validate_ahead, 1.0, max(v_ref, 1.0)))
        horizon = self.horizon(max(v_plan, abs(v)))
        s_now = float(ref.s[ref_idx])
        pose_map = pose if frame is None else compose(ominus(frame, np.zeros(3)), pose)
        d0 = ref.lateral_offset(ref_idx, pose_map)
        s_grid = np.arange(0.0, horizon + cfg.lookahead_min + cfg.lookahead_gain * v_plan + 10.0, cfg.ds)
        ref_xy, ref_yaw = ref.sample(s_now + s_grid)
        if frame is not None:
            ref_xy, ref_yaw = transform_points(ref_xy, frame), ref_yaw + frame[2]
        normal = np.column_stack([-np.sin(ref_yaw), np.cos(ref_yaw)])

        # The committed lateral manoeuvre is anchored to the route's arc length so re-planning
        # continues it instead of restarting it (which would postpone every swerve).
        prof = self.profile
        d_start = d0
        if prof is not None:
            val = self._lat_profile(prof, self._route_rel(ref, s_now, prof[0]))
            if abs(val - d0) < cfg.profile_tolerance:
                d_start = float(val)
            else:
                prof = None

        cands = []
        for off in cfg.offsets:
            if prof is not None and off == prof[2]:
                params = prof
            else:
                dd = off - d_start
                trans = max(cfg.min_transition, v_plan * math.sqrt(6.0 * abs(dd) / cfg.a_lat_max))
                params = (s_now, d_start, off, trans)
            lat = self._lat_profile(params, self._route_rel(ref, s_now + s_grid, params[0]))
            path_xy = ref_xy + lat[:, None] * normal
            valid = True
            if self.drivable is not None and off != 0.0:   # the route itself is always drivable
                pts = path_xy[: int(horizon / cfg.ds):4]
                if frame is not None:
                    pts = transform_points(pts, ominus(frame, np.zeros(3)))   # drivable() takes map coords
                valid = bool(np.all(self.drivable(pts)))
            poses, s, curv = self.rollout(pose, delta, path_xy, v_plan, horizon)
            # stop-or-go: once committed to going through, keep judging at the committed speed so that
            # a moment of braking does not make "go" look worse and flip the decision back and forth
            v_go = max(abs(v), self.go_speed or 0.0)
            s_free, clear, creeping, s_static = self.free_distance(poses, s, obstacles, v_plan, v_now=v_go)
            go_clear = self.last_go_clear
            # curvature over the stretch needed to stop: limits speed via lateral acceleration
            near = s <= max(self.stopping_distance(v_plan), 5.0)
            k_max = float(np.abs(curv[near]).max()) if near.any() else 0.0
            v_curve = math.sqrt(cfg.a_lat_max / k_max) if k_max > 1e-6 else math.inf
            v_safe = min(v_plan, v_curve, self.v_from_distance(s_free - cfg.stop_buffer))
            if creeping:
                v_safe = min(v_safe, cfg.creep_speed)
            c = Candidate(off, path_xy, poses, s, s_free, clear, v_safe, valid=valid)
            c.profile, c.s_static = params, s_static
            d_stop_now = self.stopping_distance(abs(v), self.p.max_decel) + cfg.stop_buffer
            c.go = False
            if go_clear and s_free < d_stop_now:
                # Too late to stop short of a moving obstacle's path, but keeping the current speed
                # clears it in time: go through instead of stopping in its way.
                c.s_free, c.go = s_static, True
                c.v_safe = min(v_go, v_curve, self.v_from_distance(s_static - cfg.stop_buffer))
            cands.append(c)

        for c in cands:
            if not c.valid:
                continue
            # penalise passes that use up the speed-dependent margin and little more
            spare = c.clearance - self.margin(v_plan)
            clear_pen = math.exp(-max(spare, 0.0) / 0.5) if math.isfinite(spare) else 0.0
            switch = 0.0 if self.prev_offset is None else cfg.w_switch * abs(c.offset - self.prev_offset)
            c.cost = (cfg.w_offset * abs(c.offset) + switch + (cfg.w_tabu if c.offset in self.tabu else 0.0)
                      + cfg.w_clear * clear_pen + cfg.w_speed * (v_ref - c.v_safe) / max(v_ref, 0.5)
                      + cfg.w_block * max(0.0, 1.0 - c.s_free / horizon))
        valid = [c for c in cands if c.valid] or cands
        # Only candidates the robot can still stop on before their first conflict, braking as hard as
        # it can from its *current* speed, are eligible: switching onto a path it cannot stop on in
        # time is never an improvement.
        d_emergency = self.stopping_distance(abs(v), self.p.max_decel) + cfg.stop_buffer
        moving = [c for c in valid if c.v_safe > 0.05 and (c.s_free >= d_emergency or abs(v) < 0.3)]
        if moving:
            best = min(moving, key=lambda c: c.cost)
        else:
            # Blocked or emergency: brake along the candidate with the most room before *static*
            # obstacles (certain) rather than predicted ones (a guess); ties keep the committed
            # manoeuvre so the robot never swerves back into what it was avoiding.
            prev = 0.0 if self.prev_offset is None else self.prev_offset
            best = max(valid, key=lambda c: (min(c.s_static, c.s_free + d_emergency), c.s_free,
                                             -abs(c.offset - prev)))
        blocked = not moving
        plan = Plan(Path(best.path_xy), best.offset, 0.0 if blocked else best.v_safe,
                    best.s_free, best.clearance, blocked, cands)
        plan.cost = best.cost if not blocked else math.inf
        if commit:
            self.go_speed = max(self.go_speed or 0.0, abs(v)) if getattr(best, "go", False) and not blocked else None
            self.prev_offset = best.offset
            self.profile = best.profile
            self.tabu = {k: n - 1 for k, n in self.tabu.items() if n > 1}
            self.last = plan
        return plan

    @staticmethod
    def _lat_profile(params, s_rel):
        """Smooth-step lateral offset: ``d_from`` -> ``d_to`` over ``length`` metres from the anchor."""
        _, d_from, d_to, length = params
        t = np.clip(np.asarray(s_rel) / length, 0.0, 1.0)
        return d_from + (d_to - d_from) * t * t * (3.0 - 2.0 * t)

    def straight_free_distance(self, obstacles_body: ObstacleSet, length: float, v: float) -> float:
        """Free distance straight ahead in the body frame (used before localization is ready)."""
        s = np.arange(0.0, length + self.cfg.ds, self.cfg.ds)
        poses = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
        return self.free_distance(poses, s, obstacles_body, v)[0]

    def in_moving_path(self, pose, obstacles: ObstacleSet) -> bool:
        """Is the robot (standing at ``pose``) in the predicted path of a moving obstacle?"""
        if not obstacles.dynamic:
            return False
        centres = self.footprint_centres(np.asarray(pose, dtype=float)[None])
        _, fut = obstacles.dynamic_conflicts(centres, np.zeros(1), self.circle_r + self.cfg.margin_static,
                                             self.cfg.dyn_sigma_v, self.cfg.dyn_t_max, window=0.0)
        return bool(fut[0])

    def mark_dead_end(self) -> None:
        """The current manoeuvre got the robot stuck: avoid it for a while and start afresh."""
        if self.prev_offset is not None:
            self.tabu[self.prev_offset] = self.cfg.tabu_plans
        self.prev_offset = None
        self.profile = None

    def choose_recovery(self, pose, ref: Path, ref_idx: int, v_ref: float, obstacles: ObstacleSet, frame=None):
        """Pick a back-up arc (steer, distance): try reversing left / straight / right, plan forward from
        where each ends and take the one with the best follow-up plan (an N-point-turn look-ahead)."""
        cfg = self.cfg
        best = None
        for frac in (0.0, -0.7, 0.7):
            steer = frac * self.p.max_steer
            free = self.reverse_free_distance(pose, steer, obstacles, cfg.recover_distance + cfg.stop_buffer)
            dist = min(free - cfg.stop_buffer, cfg.recover_distance)
            if dist < 0.5:
                continue
            end = np.asarray(pose, dtype=float)
            for _ in range(int(dist / cfg.ds)):
                end = bicycle_step(end, -1.0, steer, cfg.ds, self.p.wheelbase)
            end_map = end if frame is None else compose(ominus(frame, np.zeros(3)), end)
            follow = self.plan(end, 0.0, steer, ref, ref.nearest_index(end_map, ref_idx), v_ref, obstacles,
                               commit=False, frame=frame)
            score = (follow.cost, -dist)
            if best is None or score < best[0]:
                best = (score, steer, dist)
        return (0.0, 0.0) if best is None else (best[1], best[2])

    def reverse_free_distance(self, pose, steer: float, obstacles: ObstacleSet, length: float) -> float:
        """Free distance when backing up from ``pose`` with a fixed steering angle."""
        s = np.arange(0.0, length + self.cfg.ds, self.cfg.ds)
        poses = [np.asarray(pose, dtype=float)]
        for _ in s[1:]:
            poses.append(bicycle_step(poses[-1], -1.0, steer, self.cfg.ds, self.p.wheelbase))
        # static obstacles only: backing up is how the robot gets out of a walker's predicted path
        return self.free_distance(np.array(poses), s, ObstacleSet(obstacles.points), self.cfg.recover_speed)[0]

    def brake_limit(self, v: float, s_free_remaining: float) -> float | None:
        """Deceleration (negative accel) needed to stop before the obstacle, or None if not needed."""
        if v < 0.05:
            return None
        dist = s_free_remaining - self.cfg.stop_buffer
        if dist <= 0.0:
            return -math.inf
        a_need = v * v / (2.0 * dist)
        return -a_need if a_need > 0.3 * self.cfg.brake_decel else None
