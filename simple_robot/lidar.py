"""2D LiDAR processing and scan-to-map matching (point-to-line ICP).

The robot localizes against a *local* map built from its own recent scans.
Each new scan is aligned to that map with a robust point-to-line ICP seeded by
the EKF prediction; the result is an absolute pose measurement (with a
covariance from the ICP Hessian, so geometrically degenerate scenes such as
long featureless corridors produce large uncertainty along the corridor).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import rot2d, transform_points, wrap_angle


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Keep the centroid of the points in every ``voxel`` x ``voxel`` cell."""
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(counts), 2))
    np.add.at(out, inv, points)
    return out / counts[:, None]


def cloud_to_scan(points_xyz: np.ndarray, z_min: float, z_max: float,
                  r_min: float = 1.0, r_max: float = 50.0) -> np.ndarray:
    """Flatten a 3D point cloud (robot frame) into 2D points by slicing a height band."""
    z = points_xyz[:, 2]
    xy = points_xyz[:, :2]
    r = np.linalg.norm(xy, axis=1)
    m = (z > z_min) & (z < z_max) & (r > r_min) & (r < r_max)
    return xy[m]


def estimate_normals(points: np.ndarray, tree: cKDTree, k: int = 8):
    """Per-point line normals and a linearity score in [0, 1] using local PCA."""
    k = min(k, len(points))
    _, idx = tree.query(points, k=k)
    nb = points[idx] - points[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / k
    evals, evecs = np.linalg.eigh(cov)  # ascending
    normals = evecs[:, :, 0]
    linearity = 1.0 - evals[:, 0] / np.maximum(evals[:, 1], 1e-12)
    return normals, linearity


@dataclass
class ICPResult:
    pose: np.ndarray
    cov: np.ndarray
    converged: bool
    inlier_ratio: float
    rmse: float
    iterations: int


def icp_point_to_line(source: np.ndarray, target: np.ndarray, normals: np.ndarray, tree: cKDTree,
                      init_pose, max_iter: int = 30, max_corr: float = 1.5, min_corr: float = 0.3,
                      huber: float = 0.1, tol: float = 1e-4, min_points: int = 30) -> ICPResult:
    """Align body-frame ``source`` points onto world-frame ``target`` points.

    Gauss-Newton on the robust point-to-line error; the correspondence radius is
    annealed from ``max_corr`` down to ``min_corr``.
    """
    pose = np.asarray(init_pose, dtype=float).copy()
    corr = max_corr
    H = np.eye(3)
    rmse, ratio, it = np.inf, 0.0, 0
    for it in range(1, max_iter + 1):
        R = rot2d(pose[2])
        rp = source @ R.T
        world = rp + pose[:2]
        d, idx = tree.query(world, distance_upper_bound=corr)
        m = np.isfinite(d)
        if m.sum() < min_points:
            return ICPResult(pose, np.eye(3) * 1e3, False, m.mean() if len(m) else 0.0, np.inf, it)
        n = normals[idx[m]]
        r = np.einsum("ij,ij->i", n, world[m] - target[idx[m]])
        J = np.column_stack([n[:, 0], n[:, 1], n[:, 0] * -rp[m, 1] + n[:, 1] * rp[m, 0]])
        a = np.abs(r)
        w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
        H = J.T @ (J * w[:, None])
        g = J.T @ (w * r)
        dx = -np.linalg.solve(H + 1e-6 * np.eye(3), g)
        pose[:2] += dx[:2]
        pose[2] = float(wrap_angle(pose[2] + dx[2]))
        rmse = float(np.sqrt(np.mean(r ** 2)))
        ratio = float(m.mean())
        corr = max(min_corr, corr * 0.7)
        if np.linalg.norm(dx[:2]) < tol and abs(dx[2]) < tol and corr <= min_corr:
            break
    dof = max(len(r) - 3, 1)
    sigma2 = float(np.sum(w * r ** 2) / dof)
    cov = sigma2 * np.linalg.inv(H + 1e-9 * np.eye(3))
    return ICPResult(pose, cov, True, ratio, rmse, it)


@dataclass
class LidarMapperConfig:
    voxel: float = 0.25               # map / scan voxel size [m]
    map_radius: float = 40.0          # keep map points within this radius of the robot [m]
    keyframe_dist: float = 1.5        # insert scan into map after moving this far [m]
    keyframe_angle: float = np.deg2rad(10.0)
    min_linearity: float = 0.6        # only map points with a well-defined local line are used
    min_inlier_ratio: float = 0.5     # reject matches below this
    max_rmse: float = 0.25            # reject matches above this [m]
    cov_scale: float = 10.0           # ICP Hessian covariances are optimistic: inflate
    min_std_xy: float = 0.02          # covariance floor [m]
    min_std_yaw: float = np.deg2rad(0.2)


class LidarMapper:
    """Maintains a sliding local point map and matches incoming scans against it."""

    def __init__(self, cfg: LidarMapperConfig | None = None):
        self.cfg = cfg or LidarMapperConfig()
        self.points = np.zeros((0, 2))
        self.normals = np.zeros((0, 2))
        self.tree: cKDTree | None = None
        self.last_kf_pose: np.ndarray | None = None

    @property
    def empty(self) -> bool:
        return self.tree is None

    def match(self, scan_body: np.ndarray, guess) -> ICPResult | None:
        if self.empty or len(scan_body) < 30:
            return None
        src = voxel_downsample(scan_body, self.cfg.voxel)
        res = icp_point_to_line(src, self.points, self.normals, self.tree, guess)
        c = self.cfg
        if not res.converged or res.inlier_ratio < c.min_inlier_ratio or res.rmse > c.max_rmse:
            res.converged = False
            return res
        res.cov = res.cov * c.cov_scale + np.diag([c.min_std_xy ** 2, c.min_std_xy ** 2, c.min_std_yaw ** 2])
        return res

    def maybe_insert(self, scan_body: np.ndarray, pose) -> bool:
        pose = np.asarray(pose, dtype=float)
        if self.last_kf_pose is not None:
            dd = np.linalg.norm(pose[:2] - self.last_kf_pose[:2])
            da = abs(wrap_angle(pose[2] - self.last_kf_pose[2]))
            if dd < self.cfg.keyframe_dist and da < self.cfg.keyframe_angle:
                return False
        self.insert(scan_body, pose)
        return True

    def insert(self, scan_body: np.ndarray, pose) -> None:
        world = transform_points(scan_body, pose)
        pts = np.vstack([self.points, world]) if len(self.points) else world
        pts = pts[np.linalg.norm(pts - pose[:2], axis=1) < self.cfg.map_radius]
        pts = voxel_downsample(pts, self.cfg.voxel)
        if len(pts) < 10:
            return
        tree = cKDTree(pts)
        normals, lin = estimate_normals(pts, tree)
        keep = lin > self.cfg.min_linearity
        self.points, self.normals = pts[keep], normals[keep]
        self.tree = cKDTree(self.points) if len(self.points) >= 10 else None
        self.last_kf_pose = np.asarray(pose, dtype=float).copy()

    def reset(self) -> None:
        self.__init__(self.cfg)
