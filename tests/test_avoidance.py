import math
import os
import sys

import numpy as np
import pytest

from simple_robot.avoidance import AvoidanceConfig, LocalPlanner, ObstacleSet, ObstacleTracker
from simple_robot.planning import Path, resample
from simple_robot.vehicle import AckermannParams, AckermannVehicle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from run_sim2d import run  # noqa: E402

PARAMS = AckermannParams(wheelbase=2.8, track=1.6, accel_tau=0.3)
ROAD = Path(resample(np.array([[-10.0, 0.0], [200.0, 0.0]]), 0.5))


def box_points(cx, cy, w, h, step=0.1):
    xs, ys = np.arange(cx - w / 2, cx + w / 2 + 1e-9, step), np.arange(cy - h / 2, cy + h / 2 + 1e-9, step)
    return np.vstack([np.column_stack([xs, np.full_like(xs, cy - h / 2)]), np.column_stack([xs, np.full_like(xs, cy + h / 2)]),
                      np.column_stack([np.full_like(ys, cx - w / 2), ys]), np.column_stack([np.full_like(ys, cx + w / 2), ys])])


def plan(planner, v, obstacles, pose=(0.0, 0.0, 0.0), v_ref=6.0):
    idx = ROAD.nearest_index(pose)
    return planner.plan(np.array(pose), v, 0.0, ROAD, idx, v_ref, obstacles)


def test_stopping_distance_includes_reaction_and_inertia():
    lp = LocalPlanner(PARAMS)
    assert lp.reaction_time == pytest.approx(lp.cfg.reaction_time + PARAMS.accel_tau)
    for v in (0.5, 2.0, 6.0):
        d = lp.stopping_distance(v)
        assert d == pytest.approx(v * lp.reaction_time + v * v / (2 * lp.cfg.brake_decel))
        assert lp.v_from_distance(d) == pytest.approx(v)
    assert lp.stopping_distance(6.0) > 2 * lp.stopping_distance(3.0)       # grows faster than linear
    assert lp.margin(6.0) > lp.margin(1.0) > 0                              # clearance grows with speed
    assert lp.horizon(8.0) >= lp.stopping_distance(8.0)                     # always look past the stop point


def closed_loop(v_max, segs, duration=35.0, x_goal=60.0):
    """Full stack (localization initialised at the start) driving the 2D vehicle past ``segs``."""
    from simple_robot.robot import RobotStack
    from simple_robot.sim2d import footprint_clearance, raycast, sample_segments

    stack = RobotStack(PARAMS, ROAD, v_max=v_max)
    stack.loc.initialize(np.zeros(3), np.diag([1e-4, 1e-4, 1e-6]))
    car = AckermannVehicle(PARAMS)
    pts = sample_segments(segs)
    dt, min_clear, v_min_near = 0.05, np.inf, np.inf
    meas_v, meas_d, scan = 0.0, 0.0, None
    for k in range(int(duration / dt)):
        cmd = stack.step(dt, meas_v, meas_d, None, scan)
        car.step(cmd.accel, cmd.steer, dt, cmd.reverse)
        meas_v, meas_d = car.v, car.delta
        if k % 2 == 0:
            ang, r = raycast(car.pose[:2], car.pose[2], segs, 360, 40.0)
            m = np.isfinite(r)
            scan = np.column_stack([r[m] * np.cos(ang[m]), r[m] * np.sin(ang[m])])
        else:
            scan = None
        min_clear = min(min_clear, footprint_clearance(car.pose, PARAMS, pts))
        if car.pose[0] > x_goal:
            break
    return car, min_clear


def box_segs(cx, cy, w, h):
    c = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
    return [[*c[i], *c[(i + 1) % 4]] for i in range(4)]


@pytest.mark.parametrize("v_max", [3.0, 7.0])
def test_closed_loop_avoids_obstacle_at_any_speed(v_max):
    segs = np.array(box_segs(30.0, 0.3, 1.6, 2.0) + [[-10, -7, 200, -7], [-10, 7, 200, 7]])
    car, min_clear = closed_loop(v_max, segs)
    assert car.pose[0] > 60.0                       # got past
    assert min_clear > 0.3                          # never closer than ~the standstill margin


def test_swerve_length_grows_with_speed():
    """At higher speed the same lateral shift is spread over a longer distance (lateral accel limit)."""
    obs = ObstacleSet(box_points(40.0, 0.0, 1.5, 1.5))
    shifts = {}
    for v in (1.0, 6.0):
        lp = LocalPlanner(PARAMS, AvoidanceConfig(offsets=(0.0, 2.25)))
        lp.prev_offset = 2.25
        p = plan(lp, v, obs, v_ref=6.0)
        c = [c for c in p.candidates if c.offset == 2.25][0]
        shifts[v] = c.path_xy[int(5.0 / lp.cfg.ds), 1]    # lateral shift 5 m ahead
    assert shifts[1.0] > shifts[6.0] + 0.3


def test_blocked_road_stops_in_time():
    lp = LocalPlanner(PARAMS)
    wall = np.column_stack([np.full(200, 12.0), np.linspace(-10, 10, 200)])
    p = plan(lp, 5.0, ObstacleSet(wall))
    assert all(c.s_free < 12.0 for c in p.candidates)
    # whatever it picks, it can stop before the wall from the target speed
    assert lp.stopping_distance(p.v_target) <= p.s_free - lp.cfg.stop_buffer + 1e-6
    p_close = plan(lp, 0.0, ObstacleSet(wall), pose=(6.5, 0.0, 0.0))   # bumper ~1.9 m from the wall
    assert p_close.blocked and p_close.v_target == 0.0


def test_curvature_limits_speed():
    lp = LocalPlanner(PARAMS, AvoidanceConfig(a_lat_max=1.0))
    xy = np.array([[20 * math.sin(a), 20 - 20 * math.cos(a)] for a in np.linspace(0, math.pi, 200)])
    ring = Path(resample(xy, 0.5))
    p = lp.plan(np.zeros(3), 6.0, 0.0, ring, 0, 6.0, ObstacleSet())
    assert p.v_target <= math.sqrt(1.0 * 20.0) + 0.6         # v^2 / R <= a_lat (rollout cuts the corner a bit)


def test_tracker_estimates_velocity_and_ignores_static():
    tr = ObstacleTracker()
    wall = np.column_stack([np.linspace(-5, 30, 300), np.full(300, 8.0)])
    post = box_points(10.0, -6.0, 0.4, 0.4)
    obs = None
    for k in range(25):
        t = 0.1 * k
        ped = box_points(15.0, -6.0 + 1.2 * t, 0.5, 0.5)
        obs = tr.update(np.vstack([wall, post, ped]), t)
    assert len(obs.dynamic) == 1
    pts, vel = obs.dynamic[0][1], obs.dynamic[0][2]
    assert np.allclose(pts.mean(0)[0], 15.0, atol=0.3)
    assert vel == pytest.approx([0.0, 1.2], abs=0.25)
    assert len(obs.points) > len(wall)           # moving things are still obstacles where they are now


def test_static_object_seen_from_moving_robot_is_not_moving():
    """Faces of a box appear/disappear as the robot drives past; that must not look like motion."""
    tr = ObstacleTracker()
    box = box_points(10.0, -3.0, 1.2, 1.2)
    obs = None
    for k in range(40):
        rx = 0.5 * k                                 # robot driving past at 5 m/s
        vis = box[np.abs(np.arctan2(box[:, 1] + 3.0 - 0.0, box[:, 0] - 10.0)
                         - np.arctan2(0.0 - (-3.0), rx - 10.0)) < 1.2]   # roughly the faces seen from rx
        obs = tr.update(vis if len(vis) else box, 0.1 * k)
    assert not obs.dynamic


def test_crossing_pedestrian_robot_stops_short_of_its_path():
    lp = LocalPlanner(PARAMS)
    ped = box_points(14.0, -4.0, 0.5, 0.5)
    obs = ObstacleSet(ped, dynamic=[(ped, np.array([0.0, 1.2]))])     # walking towards the road
    p = plan(lp, 4.0, obs)
    front = PARAMS.length - PARAMS.rear_overhang
    # the robot plans to stop with its front bumper short of the pedestrian's crossing line
    assert p.s_free + front < 14.0 - 0.25
    assert lp.stopping_distance(p.v_target) <= p.s_free - lp.cfg.stop_buffer + 1e-6


def test_vehicle_reverse_gear():
    car = AckermannVehicle(AckermannParams())
    for _ in range(40):
        car.step(-2.0, 0.0, 0.05, reverse=True)
    assert car.v < -0.5 and car.pose[0] < 0
    car2 = AckermannVehicle(AckermannParams())
    car2.v = 1.0
    for _ in range(40):
        car2.step(-4.0, 0.0, 0.05)                    # braking in drive stops, never reverses
    assert car2.v == 0.0


def test_sim_obstacle_course_is_safe():
    _, s, _, _ = run(duration=70.0, seed=3, obstacles=True, verbose=False)
    assert s["collisions"] == 0 and s["min_clearance"] > 0.1
    assert s["laps"] > 0.75                        # past the pedestrian, parked car, crate and slalom


def test_sim_without_avoidance_crashes():
    _, s, _, _ = run(duration=40.0, seed=3, obstacles=True, avoidance=False, verbose=False)
    assert s["collisions"] > 0


def test_sim_road_blocked_backs_up_and_never_collides():
    wall = (6.0, -19.0, 0.5, 14.0)                 # across the whole south road, ahead of the start
    _, s, _, stack = run(duration=40.0, seed=0, extra_obstacles=[wall], verbose=False)
    assert s["collisions"] == 0
    assert s["blocked_s"] > 10.0
    assert s["reversing_s"] > 0.0                  # tried backing up to find another way
