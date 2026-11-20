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
