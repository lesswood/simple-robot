"""Pure-pursuit lateral control and PID longitudinal control for an Ackermann robot."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .planning import Path
from .vehicle import AckermannParams


@dataclass
class PurePursuit:
    params: AckermannParams
    lookahead_min: float = 3.0
    lookahead_gain: float = 0.8  # lookahead = min + gain * v
    _idx: int | None = field(default=None, init=False)

    def reset(self) -> None:
        self._idx = None

    def __call__(self, pose, v: float, path: Path) -> tuple[float, int, np.ndarray]:
        """Return (steering angle, nearest path index, lookahead target).

        ``pose`` is the rear-axle pose (x, y, yaw).
        """
        self._idx = path.nearest_index(pose, self._idx)
        ld = self.lookahead_min + self.lookahead_gain * abs(v)
        target, _ = path.point_at_distance(self._idx, ld)
        dx, dy = target - np.asarray(pose[:2])
        alpha = np.arctan2(dy, dx) - pose[2]
        ld_actual = max(np.hypot(dx, dy), 1e-3)
        delta = np.arctan2(2.0 * self.params.wheelbase * np.sin(alpha), ld_actual)
        delta = float(np.clip(delta, -self.params.max_steer, self.params.max_steer))
        return delta, self._idx, target


@dataclass
class PID:
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.0
    i_limit: float = 2.0
    _i: float = field(default=0.0, init=False)
    _prev: float | None = field(default=None, init=False)

    def reset(self) -> None:
        self._i, self._prev = 0.0, None

    def __call__(self, error: float, dt: float) -> float:
        self._i = float(np.clip(self._i + error * dt, -self.i_limit, self.i_limit))
        d = 0.0 if self._prev is None or dt <= 0 else (error - self._prev) / dt
        self._prev = error
        return self.kp * error + self.ki * self._i + self.kd * d
