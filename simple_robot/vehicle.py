"""Ackermann steering geometry and kinematic (bicycle) model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import wrap_angle


@dataclass
class AckermannParams:
    wheelbase: float = 2.8          # [m] front-to-rear axle distance
    track: float = 1.6              # [m] distance between left/right wheels
    max_steer: float = np.deg2rad(35.0)  # [rad] max virtual (bicycle) steering angle
    max_steer_rate: float = np.deg2rad(60.0)  # [rad/s]
    max_speed: float = 8.0          # [m/s]
    max_accel: float = 2.0          # [m/s^2]
    max_decel: float = 4.0          # [m/s^2] braking capability (friction limited, ~0.4 g)
    max_reverse_speed: float = 1.5  # [m/s]
    accel_tau: float = 0.0          # [s] drivetrain/brake lag: first-order response to accel commands
    # Footprint (for collision checking), measured from the rear axle
    length: float = 4.5             # [m] bumper to bumper
    width: float = 1.9              # [m]
    rear_overhang: float = 0.9      # [m] rear axle -> rear bumper

    @property
    def min_turn_radius(self) -> float:
        return self.wheelbase / np.tan(self.max_steer)


def ackermann_wheel_angles(delta: float, params: AckermannParams) -> tuple[float, float]:
    """Return (left, right) front wheel angles for a virtual bicycle steering angle ``delta``.

    With perfect Ackermann geometry all wheels rotate about one point on the
    rear-axle line, so the inner wheel turns more sharply than the outer one.
    Positive ``delta`` turns left.
    """
    if abs(delta) < 1e-9:
        return 0.0, 0.0
    radius = params.wheelbase / np.tan(delta)  # signed, rear-axle centre to ICR
    half = params.track / 2.0
    left = np.arctan(params.wheelbase / (radius - half))
    right = np.arctan(params.wheelbase / (radius + half))
    return float(left), float(right)


def bicycle_step(state, v: float, delta: float, dt: float, wheelbase: float) -> np.ndarray:
    """Propagate pose ``(x, y, yaw)`` of the rear-axle centre with speed ``v`` and steering ``delta``.

    Uses the exact arc integration (falls back to a straight line for tiny yaw rates).
    """
    x, y, yaw = state
    w = v * np.tan(delta) / wheelbase
    if abs(w) < 1e-6:
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
    else:
        r = v / w
        x += r * (np.sin(yaw + w * dt) - np.sin(yaw))
        y += -r * (np.cos(yaw + w * dt) - np.cos(yaw))
    return np.array([x, y, float(wrap_angle(yaw + w * dt))])


class AckermannVehicle:
    """Ground-truth vehicle used by the standalone simulator (actuator limits included)."""

    def __init__(self, params: AckermannParams, pose=(0.0, 0.0, 0.0)):
        self.p = params
        self.pose = np.asarray(pose, dtype=float)
        self.v = 0.0
        self.delta = 0.0
        self.accel = 0.0

    def step(self, accel_cmd: float, delta_cmd: float, dt: float, reverse: bool = False) -> None:
        """``accel_cmd`` is the signed longitudinal acceleration; ``reverse`` selects the reverse gear
        (speed is then <= 0, otherwise >= 0: brakes stop the vehicle, they do not reverse it)."""
        p = self.p
        delta_cmd = float(np.clip(delta_cmd, -p.max_steer, p.max_steer))
        max_d = p.max_steer_rate * dt
        self.delta += float(np.clip(delta_cmd - self.delta, -max_d, max_d))
        accel_cmd = float(np.clip(accel_cmd, -p.max_decel, p.max_accel))
        # Inertia of the drivetrain/brakes: the achieved acceleration lags the request.
        k = 1.0 if p.accel_tau <= 0 else min(1.0, dt / p.accel_tau)
        self.accel += k * (accel_cmd - self.accel)
        lo, hi = (-p.max_reverse_speed, 0.0) if reverse else (0.0, p.max_speed)
        v_new = float(np.clip(self.v + self.accel * dt, min(lo, self.v), max(hi, self.v)))
        if (reverse and v_new > 0.0 > self.v - 1e-9) or (not reverse and v_new < 0.0 < self.v + 1e-9):
            v_new = 0.0
        v_mid = 0.5 * (self.v + v_new)
        self.pose = bicycle_step(self.pose, v_mid, self.delta, dt, p.wheelbase)
        self.v = v_new
