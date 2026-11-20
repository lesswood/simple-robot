"""Small 2D geometry helpers (right-handed frame: x forward/east, y left/north)."""
from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform_points(points: np.ndarray, pose) -> np.ndarray:
    """Transform Nx2 points from a body frame into the frame where ``pose=(x, y, yaw)`` lives."""
    x, y, yaw = pose
    return points @ rot2d(yaw).T + np.array([x, y])


def compose(a, b):
    """Pose composition a (+) b."""
    ax, ay, ath = a
    bx, by, bth = b
    c, s = np.cos(ath), np.sin(ath)
    return np.array([ax + c * bx - s * by, ay + s * bx + c * by, wrap_angle(ath + bth)])


def ominus(a, b):
    """Relative pose of ``b`` expressed in the frame of ``a`` (a^-1 (+) b)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, float(wrap_angle(b[2] - a[2]))])
