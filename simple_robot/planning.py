"""Reference path representation with a curvature-limited speed profile."""
from __future__ import annotations

import numpy as np


class Path:
    def __init__(self, xy: np.ndarray, closed: bool = False):
        xy = np.asarray(xy, dtype=float)
        if closed and np.linalg.norm(xy[0] - xy[-1]) > 1e-6:
            xy = np.vstack([xy, xy[:1]])
        self.xy = xy
        self.closed = closed
        seg = np.diff(xy, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.s[-1])
        self.yaw = np.arctan2(seg[:, 1], seg[:, 0])
        self.yaw = np.append(self.yaw, self.yaw[-1])
        self.curvature = self._curvature()

    def _curvature(self) -> np.ndarray:
        n = len(self.xy)
        k = np.zeros(n)
        for i in range(1, n - 1):
            a, b, c = self.xy[i - 1], self.xy[i], self.xy[i + 1]
            ab, bc, ca = np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            denom = ab * bc * ca
            k[i] = 2.0 * cross / denom if denom > 1e-9 else 0.0
        return k

    def nearest_index(self, p, hint: int | None = None, window: int = 50) -> int:
        """Index of the closest path vertex; searches near ``hint`` if given (avoids jumping)."""
        n = len(self.xy)
        if hint is None:
            idx = np.arange(n)
        elif self.closed:
            idx = np.arange(hint - 5, hint + window) % (n - 1)  # last vertex duplicates the first
        else:
            idx = np.arange(max(0, hint - 5), min(n, hint + window))
        d = np.linalg.norm(self.xy[idx] - np.asarray(p)[:2], axis=1)
        return int(idx[np.argmin(d)])

    def point_at_distance(self, i: int, dist: float) -> tuple[np.ndarray, int]:
        """Point ``dist`` metres ahead (along the path) of vertex ``i``."""
        target = self.s[i] + dist
        if self.closed:
            target %= self.length
        target = min(target, self.length)
        j = int(np.clip(np.searchsorted(self.s, target) - 1, 0, len(self.seg_len) - 1))
        t = 0.0 if self.seg_len[j] < 1e-9 else (target - self.s[j]) / self.seg_len[j]
        return self.xy[j] + np.clip(t, 0, 1) * (self.xy[j + 1] - self.xy[j]), j

    def speed_profile(self, v_max: float, a_lat_max: float = 1.5, a_long_max: float = 1.5) -> np.ndarray:
        """Max speed per vertex limited by lateral acceleration, then smoothed for braking."""
        k = np.abs(self.curvature)
        v = np.minimum(v_max, np.sqrt(a_lat_max / np.maximum(k, 1e-6)))
        if not self.closed:
            v[-1] = 0.0
        for i in range(len(v) - 2, -1, -1):  # backward pass: be able to brake in time
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_long_max * self.seg_len[i]))
        return v

    def sample(self, s_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Points and headings at arc lengths ``s_query`` (wrapped for closed paths, clamped otherwise)."""
        s_query = np.asarray(s_query, dtype=float)
        s_query = s_query % self.length if self.closed else np.clip(s_query, 0.0, self.length)
        xy = np.column_stack([np.interp(s_query, self.s, self.xy[:, 0]), np.interp(s_query, self.s, self.xy[:, 1])])
        j = np.clip(np.searchsorted(self.s, s_query, side="right") - 1, 0, len(self.seg_len) - 1)
        return xy, self.yaw[j]

    def lateral_offset(self, i: int, p) -> float:
        """Signed lateral distance of ``p`` from the path at vertex ``i`` (positive = left)."""
        d = np.asarray(p)[:2] - self.xy[i]
        return float(-np.sin(self.yaw[i]) * d[0] + np.cos(self.yaw[i]) * d[1])

    def remaining(self, i: int) -> float:
        return float(self.length - self.s[i])


def resample(xy: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline at a constant arc-length spacing."""
    xy = np.asarray(xy, dtype=float)
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    s = np.arange(0.0, d[-1], step)
    return np.column_stack([np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])])
