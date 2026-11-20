"""The autonomy stack: localization -> obstacle-aware local planning -> path tracking -> commands.

Simulator-agnostic: both the standalone 2D simulator and the CARLA bridge feed
it sensor data and apply the :class:`Command` it returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .avoidance import AvoidanceConfig, LocalPlanner, ObstacleSet, ObstacleTracker, Plan
from .control import PID, PurePursuit
from .geometry import compose, ominus, transform_points
from .localization import Localizer, LocalizerConfig
from .planning import Path
from .vehicle import AckermannParams, bicycle_step


@dataclass
class Command:
    speed: float      # target speed [m/s]
    accel: float      # longitudinal acceleration request [m/s^2]
    steer: float      # virtual (bicycle) steering angle [rad], positive = left
    done: bool = False
    blocked: bool = False  # no collision-free way forward right now
    reverse: bool = False  # reverse gear (backing up to get unstuck)


class RobotStack:
    def __init__(self, params: AckermannParams, path: Path, loc_cfg: LocalizerConfig | None = None,
                 v_max: float = 6.0, init_speed: float = 2.0, a_lat_max: float = 1.5,
                 avoidance: AvoidanceConfig | bool | None = True, drivable=None,
                 replan_period: float = 0.2):
        """``avoidance``: ``True``/config enables the local planner, ``False``/``None`` follows the route blindly.
        ``drivable``: optional Nx2 -> bool callable restricting swerves to drivable ground."""
        self.params = params
        self.path = path
        self.loc = Localizer(loc_cfg or LocalizerConfig(wheelbase=params.wheelbase))
        self.pp = PurePursuit(params)
        self.local_pp = PurePursuit(params)
        self.speed_pid = PID(kp=1.2, ki=0.2)
        self.v_profile = path.speed_profile(min(v_max, params.max_speed), a_lat_max)
        self.init_speed = init_speed
        self.target = None
        self.idx: int | None = None
        self.pose_now: np.ndarray | None = None  # fused pose at the time of the last measurements
        self.planner: LocalPlanner | None = None
        if avoidance:
            cfg = avoidance if isinstance(avoidance, AvoidanceConfig) else AvoidanceConfig(a_lat_max=a_lat_max)
            self.planner = LocalPlanner(params, cfg, drivable)
        self.replan_period = replan_period
        self.plan: Plan | None = None
        self.obstacles = ObstacleSet()
        self.tracker = ObstacleTracker()
        self._blocked_for = 0.0
        self._progress: list = []            # (t, odom xy) history for stuck detection
        # Smooth local "odom" frame (wheel odometry only): obstacle tracking, local planning and path
        # tracking run here so GNSS corrections of the map pose never jolt the robot near obstacles.
        self.odom = np.zeros(3)
        self.map_in_odom: np.ndarray | None = None
        self._recovery: dict | None = None   # active back-up manoeuvre
        self.t = 0.0
        self._since_plan = (0.0, 0.0)   # (time, distance) since the last plan
        self._init_free = np.inf

    def step(self, dt: float, v_meas: float, delta_meas: float, gnss_enu=None,
             scan_body: np.ndarray | None = None, obstacles_body: np.ndarray | None = None) -> Command:
        """One control cycle.

        All arguments are measurements taken *now* (time t_k).  They are fused
        into the state at t_k, the command is computed from that pose, and the
        filter is then predicted forward to t_k + dt.  ``obstacles_body`` are
        2D obstacle points in the body frame (defaults to the LiDAR scan).
        """
        if gnss_enu is not None:
            self.loc.on_gnss(gnss_enu)
        if scan_body is not None:
            self.loc.on_lidar(scan_body)
        obs = obstacles_body if obstacles_body is not None else scan_body

        self.pose_now = self.loc.pose
        if not self.loc.initialized:
            cmd = self._init_step(v_meas, dt, obs)
        else:
            cmd = self._track_step(v_meas, delta_meas, dt, obs)
        t, s = self._since_plan
        self._since_plan = (t + dt, s + abs(v_meas) * dt)
        self.t += dt
        self.loc.predict(v_meas, delta_meas, dt)
        self.odom = bicycle_step(self.odom, v_meas, delta_meas, dt, self.params.wheelbase)
        return cmd

    # --------------------------------------------------------------------
    def _init_step(self, v: float, dt: float, obs) -> Command:
        """No heading yet: roll straight ahead slowly until GNSS gives a baseline - unless something is in the way."""
        v_ref, brake = self.init_speed, None
        if self.planner is not None:
            if obs is not None:
                self._init_free = self.planner.straight_free_distance(ObstacleSet(obs), self.planner.horizon(v_ref), v)
                self._since_plan = (0.0, 0.0)
            remaining = self._init_free - self._since_plan[1]
            v_ref = min(v_ref, self.planner.v_from_distance(remaining - self.planner.cfg.stop_buffer))
            brake = self.planner.brake_limit(v, remaining)
        return self._cmd(v_ref, 0.0, v, dt, brake, blocked=v_ref <= 0.0)

    def _track_step(self, v: float, delta: float, dt: float, obs) -> Command:
        pose_map = self.loc.pose
        self.idx = self.path.nearest_index(pose_map, self.idx)
        if not self.path.closed and self.path.remaining(self.idx) < 1.0:
            return Command(0.0, -self.params.max_decel, 0.0, done=True)
        v_ref = float(self.v_profile[self.idx])
        if self.plan is None and self.planner is not None:
            self.speed_pid.reset()

        if self.planner is None:
            steer, _, self.target = self.pp(pose_map, v, self.path)
            return self._cmd(v_ref, steer, v, dt)

        pose = self.odom                                            # local planning frame
        self.map_in_odom = compose(pose, ominus(pose_map, np.zeros(3)))
        if obs is not None:
            self.obstacles = self.tracker.update(transform_points(obs, pose), self.t)
        if self._recovery is not None:
            cmd = self._recover_step(pose, v, dt)
            if cmd is not None:
                return cmd
        if obs is not None or self.plan is None or self._since_plan[0] >= self.replan_period:
            self.plan = self.planner.plan(pose, v, delta, self.path, self.idx, v_ref, self.obstacles,
                                          frame=self.map_in_odom)
            self.local_pp.reset()
            self._since_plan = (0.0, 0.0)
        plan = self.plan
        # Stuck (an Ackermann vehicle cannot turn on the spot): back up a little and re-plan.
        # "Stuck" = hardly any progress for a while (robust to plans that flicker between blocked
        # and creeping): an Ackermann vehicle cannot turn on the spot, so back up and re-plan.
        if plan.blocked and abs(v) < 0.5 and self.planner.in_moving_path(pose, self.obstacles):
            # Standing in a pedestrian's way with no way forward: back out of it right away.
            self._recovery = dict(start=np.asarray(pose[:2]).copy(), steer=0.0, t=0.0, evade=True,
                                  dist=self.planner.cfg.recover_distance)
            cmd = self._recover_step(pose, v, dt)
            if cmd is not None:
                return cmd
        elif self._stuck(pose, dt):
            self._start_recovery(pose)
            cmd = self._recover_step(pose, v, dt)
            if cmd is not None:
                return cmd
        steer, _, self.target = self.local_pp(pose, v, plan.path)
        remaining = plan.s_free - self._since_plan[1]
        v_target = min(plan.v_target, self.planner.v_from_distance(remaining - self.planner.cfg.stop_buffer))
        return self._cmd(v_target, steer, v, dt, self.planner.brake_limit(v, remaining), plan.blocked)

    def _stuck(self, pose, dt: float) -> bool:
        cfg = self.planner.cfg
        # Waiting for a moving obstacle (e.g. a pedestrian crossing) is not being stuck.
        if any(np.min(np.linalg.norm(p - pose[:2], axis=1)) < 10.0 for _, p, _ in self.obstacles.dynamic):
            self._progress.clear()
            return False
        self._progress.append((self.t, np.asarray(pose[:2]).copy()))
        while self._progress and self.t - self._progress[0][0] > cfg.recover_after:
            self._progress.pop(0)
        self._blocked_for = self.t - self._progress[0][0]
        moved = max(np.linalg.norm(p - pose[:2]) for _, p in self._progress)
        return self._blocked_for >= cfg.recover_after - dt and moved < cfg.stuck_distance

    def _start_recovery(self, pose) -> None:
        self.planner.mark_dead_end()
        steer, dist = self.planner.choose_recovery(pose, self.path, self.idx, float(self.v_profile[self.idx]),
                                                   self.obstacles, frame=self.map_in_odom)
        self._recovery = dict(start=np.asarray(pose[:2]).copy(), steer=steer, dist=dist, t=0.0)
        self._progress.clear()

    def _recover_step(self, pose, v: float, dt: float) -> Command | None:
        """Back up along a fixed-steer arc while the space behind is free; None when finished."""
        rec, cfg = self._recovery, self.planner.cfg
        rec["t"] += dt
        backed = float(np.linalg.norm(pose[:2] - rec["start"]))
        free = self.planner.reverse_free_distance(pose, rec["steer"], self.obstacles, cfg.recover_distance)
        out_of_way = rec.get("evade") and backed > 0.5 and not self.planner.in_moving_path(pose, self.obstacles)
        if out_of_way or backed >= rec["dist"] or free <= cfg.stop_buffer or rec["t"] > 3 * cfg.recover_distance / cfg.recover_speed:
            if abs(v) > 0.05:      # come to a stop before switching back to forward
                return Command(0.0, float(np.clip(-v * 3.0, -self.params.max_decel, self.params.max_decel)),
                               rec["steer"], reverse=True)
            self._recovery = None
            self.plan = None
            self._progress.clear()
            return None
        speed = self.params.max_reverse_speed if rec.get("evade") else cfg.recover_speed
        v_ref = -min(speed, self.planner.v_from_distance(free - cfg.stop_buffer))
        accel = float(np.clip(1.5 * (v_ref - v), -self.params.max_decel, self.params.max_decel))
        return Command(v_ref, accel, rec["steer"], reverse=True)

    def _cmd(self, v_ref: float, steer: float, v_meas: float, dt: float,
             brake: float | None = None, blocked: bool = False) -> Command:
        accel = self.speed_pid(v_ref - v_meas, dt)
        if brake is not None:
            accel = min(accel, brake)  # inertia: brake at least as hard as needed to stop in time
        accel = float(np.clip(accel, -self.params.max_decel, self.params.max_accel))
        return Command(v_ref, accel, steer, blocked=blocked)
