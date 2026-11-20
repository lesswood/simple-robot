#!/usr/bin/env python3
"""Drive the Ackermann robot in CARLA using GNSS + LiDAR localization.

Start a CARLA server first (e.g. ``./CarlaUE4.sh -quality-level=Low``), then:

    python scripts/run_carla.py --town Town03 --route-length 300 --plot run.png

The robot spawns at a map spawn point, rolls forward slowly until GNSS gives
it a heading, then follows a route along the lane graph.  A local planner
avoids obstacles seen by the LiDAR with speed- and inertia-dependent safety
margins; ``--parked-cars`` and ``--walkers`` put obstacles on the route.
Blue dots in the CARLA window are the reference route, red dots the estimated
pose, green dots the local plan.  A CSV log (ground truth vs estimate) is
written for later analysis.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simple_robot.carla_bridge import CarlaConfig, CarlaRobot  # noqa: E402
from simple_robot.geometry import wrap_angle  # noqa: E402
from simple_robot.localization import LocalizerConfig  # noqa: E402
from simple_robot.planning import Path, resample  # noqa: E402
from simple_robot.robot import RobotStack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default=None, help="map to load, e.g. Town03 (default: keep current)")
    ap.add_argument("--blueprint", default="vehicle.tesla.model3")
    ap.add_argument("--spawn", type=int, default=0, help="spawn point index")
    ap.add_argument("--route-length", type=float, default=300.0)
    ap.add_argument("--v-max", type=float, default=6.0, help="max speed [m/s]")
    ap.add_argument("--control", choices=["ackermann", "pid"], default="ackermann")
    ap.add_argument("--gnss-std", type=float, default=0.5)
    ap.add_argument("--no-lidar", action="store_true", help="no LiDAR for localization (still used for avoidance)")
    ap.add_argument("--no-avoidance", action="store_true", help="follow the route blindly")
    ap.add_argument("--parked-cars", type=int, default=0, help="park N cars along the route")
    ap.add_argument("--walkers", type=int, default=0, help="N pedestrians pacing across the route")
    ap.add_argument("--timeout", type=float, default=300.0, help="max simulated time [s]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default="carla_run.csv")
    ap.add_argument("--plot", default=None, help="save a PNG of the run")
    ap.add_argument("--no-render-debug", action="store_true")
    a = ap.parse_args()

    cfg = CarlaConfig(host=a.host, port=a.port, town=a.town, blueprint=a.blueprint, spawn_index=a.spawn,
                      control=a.control, gnss_std=a.gnss_std, seed=a.seed)
    robot = CarlaRobot(cfg)
    rows = []
    try:
        params = robot.params
        print(f"vehicle: wheelbase={params.wheelbase:.2f} m, track={params.track:.2f} m, "
              f"max steer={np.rad2deg(params.max_steer):.1f} deg, rear axle at {robot.rear_axle_offset:+.2f} m")
        path = Path(resample(robot.route(a.route_length, seed=a.seed), 0.5))
        if not a.no_render_debug:
            robot.draw_path(path.xy, life_time=a.timeout)
        loc_cfg = LocalizerConfig(wheelbase=params.wheelbase, gnss_std=cfg.gnss_std,
                                  gnss_lever_arm=robot.gnss_lever_arm, use_lidar=not a.no_lidar)
        stack = RobotStack(params, path, loc_cfg, v_max=a.v_max, avoidance=not a.no_avoidance,
                           drivable=robot.drivable)
        if a.parked_cars:
            print(f"parked cars: {robot.spawn_parked_cars(path.xy, a.parked_cars)}")
        if a.walkers:
            print(f"walkers: {robot.spawn_walkers(path.xy, a.walkers)}")

        meas = robot.tick()
        t0 = time.time()
        for k in range(int(a.timeout / cfg.dt)):
            cmd = stack.step(cfg.dt, meas["v"], meas["steer"], meas["gnss"], meas["scan"], meas["obstacles"])
            robot.apply(cmd.speed, cmd.accel, cmd.steer, stop=cmd.done, reverse=cmd.reverse)
            gt = robot.ground_truth()
            est = stack.pose_now if stack.pose_now is not None else np.full(3, np.nan)
            g = meas["gnss"][:2] if meas["gnss"] is not None else (np.nan, np.nan)
            rows.append([k * cfg.dt, *gt, *est, *g, meas["v"], cmd.steer])
            if k % 10 == 0:
                robot.follow_with_spectator()
                if stack.loc.initialized and not a.no_render_debug:
                    robot.draw_estimate(est)
                    if stack.plan is not None and stack.map_in_odom is not None:
                        robot.draw_local_plan(stack.plan.path.xy, stack.map_in_odom)
            if k % 40 == 0:
                err = np.linalg.norm(gt[:2] - est[:2]) if stack.loc.initialized else float("nan")
                prog = path.s[stack.idx] / path.length * 100 if stack.idx is not None else 0.0
                state = "REVERSE" if cmd.reverse else ("BLOCKED" if cmd.blocked else "")
                print(f"t={k * cfg.dt:6.1f}s v={meas['v']:4.1f} m/s  pos err={err:5.2f} m  "
                      f"progress={prog:5.1f}%  collisions={len(robot.collisions)} {state}")
            if cmd.done and abs(meas["v"]) < 0.1:
                print("goal reached")
                break
            meas = robot.tick()
        print(f"wall time {time.time() - t0:.1f}s, collisions: {len(robot.collisions)} "
              f"{sorted(set(c[1] for c in robot.collisions))}")
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        robot.destroy()

    if not rows:
        return
    log = np.array(rows)
    with open(a.log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "gt_x", "gt_y", "gt_yaw", "est_x", "est_y", "est_yaw", "gnss_x", "gnss_y", "v", "steer"])
        w.writerows(log)
    ok = ~np.isnan(log[:, 4])
    if ok.any():
        e = np.linalg.norm(log[ok, 1:3] - log[ok, 4:6], axis=1)
        ye = np.rad2deg(np.abs(wrap_angle(log[ok, 3] - log[ok, 6])))
        print(f"position RMSE {np.sqrt(np.mean(e ** 2)):.3f} m (max {e.max():.2f} m), "
              f"yaw RMSE {np.sqrt(np.mean(ye ** 2)):.2f} deg; log -> {a.log}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.plot(*path.xy.T, "--", color="0.6", label="route")
        ax.plot(log[:, 7], log[:, 8], ".", ms=2, color="tab:orange", label="GNSS")
        ax.plot(log[:, 1], log[:, 2], color="tab:green", lw=2, label="ground truth")
        ax.plot(log[:, 4], log[:, 5], color="tab:blue", lw=1, label="EKF estimate")
        ax.set_aspect("equal")
        ax.set_xlabel("east [m]")
        ax.set_ylabel("north [m]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120)
        print(f"saved {a.plot}")


if __name__ == "__main__":
    main()
