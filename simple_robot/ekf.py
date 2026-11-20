"""Extended Kalman filter on SE(2) for an Ackermann robot, with stochastic cloning.

State: ``[x, y, yaw]`` of the rear-axle centre in the local ENU frame.
Inputs (prediction): measured wheel speed ``v`` and steering angle ``delta``.
Measurements:

* GNSS position (absolute, with an antenna lever arm);
* LiDAR odometry - the *relative* motion between two scans.  To fuse a
  relative measurement correctly the filter keeps a copy ("clone") of the pose
  at the previous scan in an augmented state ``[x_now, x_clone]`` so the
  correlation between the two poses is tracked (stochastic cloning).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import ominus, wrap_angle
from .vehicle import bicycle_step

CHI2_99 = {2: 9.21, 3: 11.34}


@dataclass
class EKFNoise:
    sigma_v: float = 0.1                    # wheel speed noise [m/s]
    sigma_v_rel: float = 0.03               # plus 3 % of speed (slip, tyre radius error)
    sigma_delta: float = np.deg2rad(1.5)    # steering angle noise [rad]
    q_xy: float = 0.02                      # extra random walk [m^2/s]
    q_yaw: float = np.deg2rad(0.5) ** 2     # extra random walk [rad^2/s]


class EKF:
    def __init__(self, x0, P0, wheelbase: float, noise: EKFNoise | None = None):
        self.X = np.asarray(x0, dtype=float).copy()   # 3 or 6 (with clone)
        self.P = np.asarray(P0, dtype=float).copy()
        self.L = wheelbase
        self.n = noise or EKFNoise()

    # -------------------------------------------------------------- accessors
    @property
    def x(self) -> np.ndarray:
        return self.X[:3]

    @property
    def Pxx(self) -> np.ndarray:
        return self.P[:3, :3]

    @property
    def has_clone(self) -> bool:
        return len(self.X) == 6

    def clone(self) -> None:
        """Augment the state with a copy of the current pose."""
        x, P = self.X[:3].copy(), self.P[:3, :3].copy()
        self.X = np.concatenate([x, x])
        self.P = np.block([[P, P], [P, P]])

    def drop_clone(self) -> None:
        self.X, self.P = self.X[:3].copy(), self.P[:3, :3].copy()

    def motion_since_clone(self) -> np.ndarray:
        return ominus(self.X[3:], self.X[:3]) if self.has_clone else np.zeros(3)

    # ------------------------------------------------------------- prediction
    def predict(self, v: float, delta: float, dt: float) -> None:
        if dt <= 0:
            return
        _, _, yaw = self.X[:3]
        L = self.L
        w = v * np.tan(delta) / L
        ym = yaw + 0.5 * w * dt  # mid-point heading for linearisation
        F = np.eye(len(self.X))
        F[0, 2] = -v * dt * np.sin(ym)
        F[1, 2] = v * dt * np.cos(ym)
        dw_dv = np.tan(delta) / L
        dw_dd = v / (L * np.cos(delta) ** 2)
        h = 0.5 * dt * v * dt
        G = np.zeros((len(self.X), 2))
        G[:3] = [
            [dt * np.cos(ym) - h * np.sin(ym) * dw_dv, -h * np.sin(ym) * dw_dd],
            [dt * np.sin(ym) + h * np.cos(ym) * dw_dv, h * np.cos(ym) * dw_dd],
            [dt * dw_dv, dt * dw_dd],
        ]
        sv = self.n.sigma_v + self.n.sigma_v_rel * abs(v)
        M = np.diag([sv ** 2, self.n.sigma_delta ** 2])
        Q = np.zeros_like(self.P)
        Q[:3, :3] = np.diag([self.n.q_xy, self.n.q_xy, self.n.q_yaw]) * dt
        self.X[:3] = bicycle_step(self.X[:3], v, delta, dt, L)
        self.P = F @ self.P @ F.T + G @ M @ G.T + Q

    # ---------------------------------------------------------------- updates
    def _update(self, innov: np.ndarray, H: np.ndarray, R: np.ndarray, gate: bool = True) -> bool:
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.inv(S)
        d2 = float(innov @ Sinv @ innov)
        if gate and d2 > CHI2_99[len(innov)]:
            return False
        K = self.P @ H.T @ Sinv
        self.X = self.X + K @ innov
        self.X[2] = float(wrap_angle(self.X[2]))
        if self.has_clone:
            self.X[5] = float(wrap_angle(self.X[5]))
        I_KH = np.eye(len(self.X)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def update_gnss(self, z_xy, R: np.ndarray, lever_arm=(0.0, 0.0), gate: bool = True) -> bool:
        """GNSS antenna at ``lever_arm`` (body frame) measures position ``z_xy``."""
        x, y, yaw = self.X[:3]
        lx, ly = lever_arm
        c, s = np.cos(yaw), np.sin(yaw)
        h = np.array([x + c * lx - s * ly, y + s * lx + c * ly])
        H = np.zeros((2, len(self.X)))
        H[:, :3] = [[1, 0, -s * lx - c * ly],
                    [0, 1, c * lx - s * ly]]
        return self._update(np.asarray(z_xy) - h, H, R, gate)

    def update_pose(self, z_pose, R: np.ndarray, gate: bool = True) -> bool:
        """Absolute pose measurement (e.g. localisation against a prior map)."""
        innov = np.asarray(z_pose, dtype=float) - self.X[:3]
        innov[2] = float(wrap_angle(innov[2]))
        H = np.zeros((3, len(self.X)))
        H[:, :3] = np.eye(3)
        return self._update(innov, H, R, gate)

    def update_relative(self, z_delta, R: np.ndarray, gate: bool = True) -> bool:
        """Relative motion ``clone^-1 (+) now`` measured in the clone's body frame."""
        if not self.has_clone:
            raise RuntimeError("update_relative needs a cloned state")
        xc, xn = self.X[3:], self.X[:3]
        h = ominus(xc, xn)
        c, s = np.cos(xc[2]), np.sin(xc[2])
        dx, dy = xn[0] - xc[0], xn[1] - xc[1]
        RT = np.array([[c, s], [-s, c]])
        H = np.zeros((3, 6))
        H[:2, :2] = RT
        H[2, 2] = 1.0
        H[:2, 3:5] = -RT
        H[:2, 5] = [-s * dx + c * dy, -c * dx - s * dy]
        H[2, 5] = -1.0
        innov = np.asarray(z_delta, dtype=float) - h
        innov[2] = float(wrap_angle(innov[2]))
        return self._update(innov, H, R, gate)
