import os
import subprocess
import sys

import numpy as np
import pytest

import tests.fake_carla as fake_carla
from simple_robot.carla_bridge import (carla_lidar_to_body, carla_xy_to_enu, carla_yaw_to_enu,
                                       enu_to_carla_xy, metres_to_lat_lon_deg)

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_conversions():
    assert np.allclose(carla_xy_to_enu(3, 4), [3, -4])
    assert enu_to_carla_xy(*carla_xy_to_enu(3, 4)) == (3, 4)
    assert carla_yaw_to_enu(90.0) == pytest.approx(-np.pi / 2)  # CARLA +90 deg faces +y (south)
    raw = np.array([[1.0, 2.0, -1.0, 0.5]], dtype=np.float32)
    assert np.allclose(carla_lidar_to_body(raw, (0.5, 0.0, 2.0)), [[1.5, -2.0, 1.0]])
    slat, slon = metres_to_lat_lon_deg(1.0, 60.0)
    assert slon == pytest.approx(2 * slat)


@pytest.fixture
def robot(monkeypatch):
    monkeypatch.setitem(sys.modules, "carla", fake_carla)
    from simple_robot.carla_bridge import CarlaConfig, CarlaRobot
    r = CarlaRobot(CarlaConfig(gnss_std=0.0, speed_noise=0.0, steer_noise_deg=0.0))
    yield r
    r.destroy()


def test_bridge_geometry_and_gnss_frame(robot):
    assert robot.params.wheelbase == pytest.approx(2.8)
    assert robot.rear_axle_offset == pytest.approx(-1.4)
    meas = None
    for _ in range(4):
        meas = robot.tick()
        if meas["gnss"] is not None:
            break
    # GNSS (antenna at actor origin = 1.4 m ahead of rear axle) agrees with ground truth in ENU
    gt = robot.ground_truth()
    ant = gt[:2] + 1.4 * np.array([np.cos(gt[2]), np.sin(gt[2])])
    assert np.allclose(meas["gnss"][:2], ant, atol=1e-6)
    assert meas["scan"] is not None and len(meas["scan"]) > 100
    # walls at CARLA y=+-9 -> ENU y=-+9, relative to the vehicle on the road centre
    assert np.isclose(np.abs(meas["scan"][:, 1]).max(), 9.0, atol=0.3)


def test_bridge_steering_sign(robot):
    # Command a left turn (positive in ENU) -> CARLA yaw must decrease (counter-clockwise from above)
    for _ in range(40):
        robot.tick()
        robot.apply(3.0, 1.0, np.deg2rad(15))
    assert robot.ground_truth()[2] > 0.05
    m = robot.tick()
    assert m["steer"] == pytest.approx(np.deg2rad(15), abs=1e-6)


def test_run_carla_script_end_to_end(tmp_path):
    """Run scripts/run_carla.py against the fake CARLA module."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "tests"), ROOT]))
    shim = tmp_path / "carla.py"
    shim.write_text("from fake_carla import *\nfrom fake_carla import command\n")
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), env["PYTHONPATH"]])
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_carla.py"), "--route-length", "120",
                          "--timeout", "60", "--log", str(tmp_path / "log.csv"), "--plot", str(tmp_path / "p.png")],
                         env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "goal reached" in out.stdout, out.stdout
    rmse = float(out.stdout.split("position RMSE ")[1].split(" m")[0])
    assert rmse < 0.1  # fake GNSS is noise-free, out.stdout


def test_remove_own_body():
    from simple_robot.carla_bridge import remove_own_body
    from simple_robot.vehicle import AckermannParams
    p = AckermannParams(length=4.6, width=1.9, rear_overhang=0.9)
    pts = np.array([[0.0, 0.0, 1.0], [3.5, 0.5, 1.0], [-1.0, 0.0, 1.0], [5.0, 0.0, 1.0], [1.0, 2.0, 1.0]])
    assert np.allclose(remove_own_body(pts, p), pts[[3, 4]])


def test_bridge_footprint_from_bounding_box(robot):
    assert robot.params.length == pytest.approx(4.6)
    assert robot.params.width == pytest.approx(1.9)
    assert robot.params.rear_overhang == pytest.approx(0.9)   # rear axle 1.4 m behind centre, bumper 2.3 m
    assert robot.drivable(np.array([[10.0, 0.0], [10.0, 8.5]])).tolist() == [True, False]


def _run_fake(tmp_path, *extra):
    env = dict(os.environ)
    shim = tmp_path / "carla.py"
    shim.write_text("from fake_carla import *\nfrom fake_carla import command\n")
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), os.path.join(ROOT, "tests"), ROOT])
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_carla.py"), "--route-length", "200",
                           "--timeout", "90", "--log", str(tmp_path / "log.csv"), *extra],
                          env=env, capture_output=True, text=True, timeout=600)


def test_run_carla_avoids_parked_cars_and_walkers(tmp_path):
    out = _run_fake(tmp_path, "--parked-cars", "3", "--walkers", "1")
    assert out.returncode == 0, out.stderr
    assert "parked cars: 3" in out.stdout and "walkers: 1" in out.stdout
    assert "goal reached" in out.stdout, out.stdout
    assert "collisions: 0 " in out.stdout, out.stdout
    blind = _run_fake(tmp_path, "--parked-cars", "3", "--no-avoidance")
    assert "collisions: 0 " not in blind.stdout, blind.stdout
