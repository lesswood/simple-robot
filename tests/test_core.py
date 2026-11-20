import numpy as np
import pytest

from simple_robot.control import PurePursuit
from simple_robot.ekf import EKF
from simple_robot.geometry import compose, ominus, transform_points, wrap_angle
from simple_robot.gnss import CarlaGeoProjection, LocalTangentPlane
from simple_robot.lidar import LidarMapper, estimate_normals, icp_point_to_line, voxel_downsample
from simple_robot.localization import Localizer, LocalizerConfig, rigid_align_2d
from simple_robot.planning import Path, resample
from simple_robot.vehicle import AckermannParams, ackermann_wheel_angles, bicycle_step

from scipy.spatial import cKDTree


def test_ackermann_inner_wheel_turns_more():
    p = AckermannParams(wheelbase=2.8, track=1.6)
    left, right = ackermann_wheel_angles(np.deg2rad(20), p)
    assert left > np.deg2rad(20) > right > 0  # left turn: left wheel is inner
    l2, r2 = ackermann_wheel_angles(-np.deg2rad(20), p)
    assert l2 == pytest.approx(-right) and r2 == pytest.approx(-left)
    # all wheel axes meet at the instantaneous centre of rotation
    R = p.wheelbase / np.tan(np.deg2rad(20))
    assert p.wheelbase / np.tan(left) == pytest.approx(R - p.track / 2)


def test_bicycle_full_circle():
    L, delta, v = 2.5, 0.3, 2.0
    R = L / np.tan(delta)
    T = 2 * np.pi * R / v
    n = 1000
    s = np.zeros(3)
    for _ in range(n):
        s = bicycle_step(s, v, delta, T / n, L)
    assert np.allclose(s, [0, 0, 0], atol=1e-6)


def test_pose_algebra():
    a, b = np.array([1.0, 2.0, 0.5]), np.array([-0.3, 0.8, -1.0])
    assert np.allclose(compose(a, ominus(a, b)), b)
    assert wrap_angle(3 * np.pi) == pytest.approx(-np.pi)


@pytest.mark.parametrize("proj", [CarlaGeoProjection, LocalTangentPlane])
def test_geo_projection_round_trip_and_scale(proj):
    p = proj(52.0, 4.37, 0.0)
    if proj is CarlaGeoProjection:
        lat, lon, _ = p.from_enu(123.4, -56.7)
        assert np.allclose(p.to_enu(lat, lon)[:2], [123.4, -56.7], atol=1e-6)
    # ~111 km per degree of latitude, north is +y
    e, n, _ = p.to_enu(52.001, 4.37, 0.0)
    assert abs(e) < 0.05 and n == pytest.approx(111.25, abs=0.3)


def test_carla_projection_matches_libcarla_formula():
    # Re-implementation of carla::geom::GeoLocation::Transform for a CARLA location (x, y)
    lat0, lon0 = 49.0, 8.0
    R = 6378137.0
    x, y = 250.0, 75.0
    scale = np.cos(np.deg2rad(lat0))
    mx = scale * np.deg2rad(lon0) * R + x
    my = scale * R * np.log(np.tan((90 + lat0) * np.pi / 360)) - y
    lon = mx * 180 / (np.pi * R * scale)
    lat = 360 * np.arctan(np.exp(my / (R * scale))) / np.pi - 90
    enu = CarlaGeoProjection(lat0, lon0).to_enu(lat, lon)
    assert np.allclose(enu[:2], [x, -y], atol=1e-6)  # east = x, north = -y


def _room(n=400):
    t = np.linspace(0, 1, n)
    walls = [np.column_stack([t * 30 - 15, np.full(n, -8)]), np.column_stack([t * 30 - 15, np.full(n, 8)]),
             np.column_stack([np.full(n, -15), t * 16 - 8]), np.column_stack([np.full(n, 15), t * 16 - 8]),
             np.column_stack([t * 4 + 2, np.full(n, 3)]), np.column_stack([np.full(n, -5), t * 5 - 6])]
    return np.vstack(walls)


def test_icp_recovers_pose():
    world = voxel_downsample(_room(), 0.1)
    true_pose = np.array([1.0, -0.5, 0.15])
    c, s = np.cos(true_pose[2]), np.sin(true_pose[2])
    scan = (world - true_pose[:2]) @ np.array([[c, -s], [s, c]])  # world -> body
    tree = cKDTree(world)
    normals, _ = estimate_normals(world, tree)
    res = icp_point_to_line(scan, world, normals, tree, true_pose + [0.4, -0.3, -0.08])
    assert res.converged
    assert np.allclose(res.pose, true_pose, atol=1e-3)


def test_mapper_match_after_insert():
    world = _room()
    m = LidarMapper()
    m.insert(world, np.zeros(3))
    pose = np.array([0.5, 0.2, 0.05])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    scan = (world - pose[:2]) @ np.array([[c, -s], [s, c]])
    res = m.match(scan, [0.3, 0.3, 0.0])
    assert res.converged and np.allclose(res.pose, pose, atol=0.02)


def test_ekf_gnss_converges_and_lever_arm():
    ekf = EKF([0, 0, 0], np.diag([25, 25, 0.1]), 2.8)
    truth = np.array([3.0, -2.0, 0.0])
    for _ in range(50):
        ekf.update_gnss(transform_points(np.array([[1.0, 0.0]]), truth)[0], np.eye(2) * 0.25, (1.0, 0.0), gate=False)
    assert np.allclose(ekf.x[:2], truth[:2], atol=0.05)


def test_ekf_relative_update_with_clone():
    ekf = EKF([0, 0, 0], np.diag([0.01, 0.01, 0.001]), 2.8)
    ekf.clone()
    for _ in range(10):
        ekf.predict(2.0, 0.0, 0.1)          # wheel odometry says 2 m
    ekf.update_relative([1.8, 0.0, 0.0], np.diag([1e-4, 1e-4, 1e-6]))  # lidar says 1.8 m
    assert ekf.x[0] == pytest.approx(1.8, abs=0.02)
    assert ekf.has_clone and np.all(np.linalg.eigvalsh(ekf.P) > -1e-12)


def test_rigid_align():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(20, 2))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    R_est, t = rigid_align_2d(src, src @ R.T + [3, 4])
    assert np.allclose(R_est, R) and np.allclose(t, [3, 4])


def test_localizer_initializes_heading_from_gnss():
    loc = Localizer(LocalizerConfig(wheelbase=2.8, init_baseline=5.0))
    yaw = 1.0
    pose = np.array([10.0, 20.0, yaw])
    for k in range(60):
        loc.predict(2.0, 0.0, 0.1)
        pose = bicycle_step(pose, 2.0, 0.0, 0.1, 2.8)
        if k % 2 == 0:
            loc.on_gnss(pose[:2])
    assert loc.initialized
    assert abs(wrap_angle(loc.pose[2] - yaw)) < np.deg2rad(1)
    assert np.linalg.norm(loc.pose[:2] - pose[:2]) < 0.3


def test_pure_pursuit_steers_toward_path():
    p = AckermannParams()
    path = Path(resample(np.array([[0, 0], [50, 0]]), 0.5))
    pp = PurePursuit(p)
    left, _, _ = pp([10, -2, 0], 3.0, path)   # right of path -> steer left
    pp.reset()
    right, _, _ = pp([10, 2, 0], 3.0, path)
    assert left > 0 > right


def test_closed_path_wraps():
    xy = np.array([[np.cos(a) * 10, np.sin(a) * 10] for a in np.linspace(0, 2 * np.pi, 100)])
    path = Path(xy, closed=True)
    i = path.nearest_index([10, -0.3], hint=len(path.xy) - 3)
    assert i < 5 or i > len(path.xy) - 5
