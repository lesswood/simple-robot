"""GNSS + LiDAR localization: EKF fusion with automatic initialization.

Life cycle
----------
1. **Initializing** - the robot has no heading information (GNSS gives position
   only).  While it drives, wheel odometry is dead-reckoned in an arbitrary
   "odom" frame and paired with GNSS fixes.  Once the GNSS fixes span
   ``init_baseline`` metres, a 2D rigid alignment (Umeyama/Kabsch) between the
   odometry track and the GNSS track yields the initial position *and* heading.
2. **Tracking** - the EKF predicts with wheel speed + steering and corrects
   with absolute GNSS fixes and with *relative* LiDAR odometry.

LiDAR odometry runs in its own drifting "lidar-odom" frame: every scan is
matched (point-to-line ICP) against a sliding local map built only from
earlier LiDAR poses, and the motion between consecutive scans is fused in the
EKF via stochastic cloning.  Keeping the LiDAR map independent of the GNSS
corrections is what makes the two measurements statistically independent;
inserting scans at the fused pose instead would make the LiDAR "measurement"
an echo of the filter's own past estimate and drown out GNSS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ekf import EKF, EKFNoise
from .geometry import compose, ominus, wrap_angle
from .lidar import LidarMapper, LidarMapperConfig
from .vehicle import bicycle_step


@dataclass
class LocalizerConfig:
    wheelbase: float = 2.8
    gnss_std: float = 0.5                 # [m] 1-sigma horizontal GNSS noise
    gnss_lever_arm: tuple = (0.0, 0.0)    # antenna position in body frame [m]
    init_baseline: float = 6.0            # [m] GNSS spread needed to initialise heading
    use_lidar: bool = True
    use_gnss: bool = True
    ekf_noise: EKFNoise = field(default_factory=EKFNoise)
    mapper: LidarMapperConfig = field(default_factory=LidarMapperConfig)


def rigid_align_2d(src: np.ndarray, dst: np.ndarray):
    """Least-squares R, t with dst ~ R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Hm = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(Hm)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mu_d - R @ mu_s


class Localizer:
    def __init__(self, cfg: LocalizerConfig | None = None):
        self.cfg = cfg or LocalizerConfig()
        self.ekf: EKF | None = None
        self.mapper = LidarMapper(self.cfg.mapper)
        self._odom = np.zeros(3)
        self._init_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self.lidar_pose: np.ndarray | None = None   # pose in the lidar-odom frame
        self.stats = {"gnss_ok": 0, "gnss_rej": 0, "lidar_ok": 0, "lidar_rej": 0}

    # ------------------------------------------------------------------ state
    @property
    def initialized(self) -> bool:
        return self.ekf is not None

    @property
    def pose(self) -> np.ndarray | None:
        return None if self.ekf is None else self.ekf.x.copy()

    @property
    def cov(self) -> np.ndarray | None:
        return None if self.ekf is None else self.ekf.Pxx.copy()

    def initialize(self, pose, cov=None) -> None:
        """Initialise directly from a known pose (e.g. a known start position)."""
        cov = np.diag([1.0, 1.0, np.deg2rad(10) ** 2]) if cov is None else cov
        self.ekf = EKF(pose, cov, self.cfg.wheelbase, self.cfg.ekf_noise)

    # ----------------------------------------------------------------- inputs
    def predict(self, v: float, delta: float, dt: float) -> None:
        if self.ekf is None:
            self._odom = bicycle_step(self._odom, v, delta, dt, self.cfg.wheelbase)
        else:
            self.ekf.predict(v, delta, dt)

    def on_gnss(self, enu_xy) -> bool:
        z = np.asarray(enu_xy, dtype=float)[:2]
        if self.ekf is None:
            self._try_initialize(z)
            return False
        if not self.cfg.use_gnss:
            return False
        R = np.eye(2) * self.cfg.gnss_std ** 2
        ok = self.ekf.update_gnss(z, R, self.cfg.gnss_lever_arm)
        self.stats["gnss_ok" if ok else "gnss_rej"] += 1
        if not ok and np.trace(self.ekf.P[:2, :2]) < 1e-2:
            # Avoid a filter that is too confident to ever accept GNSS again.
            self.ekf.P[:2, :2] += np.eye(2) * 0.05
        return ok

    def on_lidar(self, scan_body: np.ndarray) -> bool:
        if self.ekf is None or not self.cfg.use_lidar or len(scan_body) < 30:
            return False
        ekf = self.ekf
        if self.lidar_pose is None or not ekf.has_clone:
            # First scan: anchor the lidar-odom frame at the current estimate.
            self.lidar_pose = ekf.x.copy()
            self.mapper.reset()
            self.mapper.insert(scan_body, self.lidar_pose)
            ekf.clone()
            return False
        guess = compose(self.lidar_pose, ekf.motion_since_clone())
        res = self.mapper.match(scan_body, guess)
        ok = False
        if res is not None and res.converged:
            delta = ominus(self.lidar_pose, res.pose)
            # Express the ICP covariance (lidar-odom frame) in the previous body frame.
            c, s = np.cos(self.lidar_pose[2]), np.sin(self.lidar_pose[2])
            J = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            ok = ekf.update_relative(delta, J @ res.cov @ J.T)
        self.stats["lidar_ok" if ok else "lidar_rej"] += 1
        self.lidar_pose = res.pose if ok else guess
        self.mapper.maybe_insert(scan_body, self.lidar_pose)
        ekf.drop_clone()
        ekf.clone()
        return ok

    # ----------------------------------------------------------------- init
    def _try_initialize(self, z: np.ndarray) -> None:
        lx, ly = self.cfg.gnss_lever_arm
        c, s = np.cos(self._odom[2]), np.sin(self._odom[2])
        antenna_odom = self._odom[:2] + np.array([c * lx - s * ly, s * lx + c * ly])
        self._init_pairs.append((antenna_odom, z))
        gnss = np.array([p[1] for p in self._init_pairs])
        if np.linalg.norm(gnss.max(0) - gnss.min(0)) < self.cfg.init_baseline or len(gnss) < 5:
            return
        odom = np.array([p[0] for p in self._init_pairs])
        R, t = rigid_align_2d(odom, gnss)
        theta = np.arctan2(R[1, 0], R[0, 0])
        xy = R @ self._odom[:2] + t
        yaw = float(wrap_angle(theta + self._odom[2]))
        resid = gnss - (odom @ R.T + t)
        pos_var = max(float(np.mean(resid ** 2)), self.cfg.gnss_std ** 2) * 2.0
        span = np.linalg.norm(gnss[-1] - gnss[0])
        yaw_std = np.clip(2.0 * self.cfg.gnss_std / max(span, 1e-3), np.deg2rad(2), np.deg2rad(30))
        self.ekf = EKF([xy[0], xy[1], yaw], np.diag([pos_var, pos_var, yaw_std ** 2]),
                       self.cfg.wheelbase, self.cfg.ekf_noise)
        self._init_pairs.clear()
