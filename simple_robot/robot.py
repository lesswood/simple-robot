"""The autonomy stack: localization -> path tracking -> actuator commands.

Simulator-agnostic: both the standalone 2D simulator and the CARLA bridge feed
it sensor data and apply the :class:`Command` it returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control import PID, PurePursuit
from .localization import Localizer, LocalizerConfig
from .planning import Path
from .vehicle import AckermannParams


@dataclass
class Command:
    speed: float      # target speed [m/s]
    accel: float      # longitudinal acceleration request [m/s^2]
    steer: float      # virtual (bicycle) steering angle [rad], positive = left
    done: bool = False


class RobotStack:
    def __init__(self, params: AckermannParams, path: Path, loc_cfg: LocalizerConfig | None = None,
                 v_max: float = 6.0, init_speed: float = 2.0, a_lat_max: float = 1.5):
        self.params = params
        self.path = path
        self.loc = Localizer(loc_cfg or LocalizerConfig(wheelbase=params.wheelbase))
        self.pp = PurePursuit(params)
        self.speed_pid = PID(kp=1.2, ki=0.2)
        self.v_profile = path.speed_profile(min(v_max, params.max_speed), a_lat_max)
        self.init_speed = init_speed
        self.target = None
        self.idx = 0
        self.pose_now: np.ndarray | None = None  # fused pose at the time of the last measurements

    def step(self, dt: float, v_meas: float, delta_meas: float,
             gnss_enu=None, scan_body: np.ndarray | None = None) -> Command:
        """One control cycle.

        All arguments are measurements taken *now* (time t_k).  They are fused
        into the state at t_k, the command is computed from that pose, and the
        filter is then predicted forward to t_k + dt.
        """
        if gnss_enu is not None:
            self.loc.on_gnss(gnss_enu)
        if scan_body is not None:
            self.loc.on_lidar(scan_body)

        self.pose_now = self.loc.pose
        if not self.loc.initialized:
            # No heading yet: roll straight ahead slowly until GNSS gives a baseline.
            cmd = self._cmd(self.init_speed, 0.0, v_meas, dt)
        else:
            steer, self.idx, self.target = self.pp(self.loc.pose, v_meas, self.path)
            if not self.path.closed and self.path.remaining(self.idx) < 1.0:
                cmd = Command(0.0, -self.params.max_decel, 0.0, done=True)
            else:
                cmd = self._cmd(float(self.v_profile[self.idx]), steer, v_meas, dt)
        self.loc.predict(v_meas, delta_meas, dt)
        return cmd

    def _cmd(self, v_ref: float, steer: float, v_meas: float, dt: float) -> Command:
        accel = float(np.clip(self.speed_pid(v_ref - v_meas, dt), -self.params.max_decel, self.params.max_accel))
        return Command(v_ref, accel, steer)
