#!/usr/bin/env python3
"""Run the full stack in the standalone 2D simulator (no CARLA required).

    python scripts/run_sim2d.py --laps 1 --plot sim2d.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simple_robot.geometry import compose, ominus, transform_points, wrap_angle  # noqa: E402
from simple_robot.localization import LocalizerConfig  # noqa: E402
from simple_robot.planning import Path  # noqa: E402
from simple_robot.robot import RobotStack  # noqa: E402
from simple_robot.sim2d import Sim2D, SimConfig  # noqa: E402
from simple_robot.vehicle import AckermannParams  # noqa: E402


def run(duration: float = 90.0, seed: int = 0, use_lidar: bool = True, use_gnss: bool = True,
        gnss_std: float = 0.5, v_max: float = 6.0, gnss_outage=None, verbose: bool = True):
    params = AckermannParams(wheelbase=2.8, track=1.6)
    cfg = SimConfig(gnss_std=gnss_std, gnss_outage=gnss_outage)
    sim = Sim2D(params, cfg, seed=seed)
    path = Path(sim.path, closed=True)
    loc_cfg = LocalizerConfig(wheelbase=params.wheelbase, gnss_std=gnss_std, gnss_lever_arm=cfg.lever_arm,
                              use_lidar=use_lidar, use_gnss=use_gnss)
    stack = RobotStack(params, path, loc_cfg, v_max=v_max)

    log = []
    meas = {"v": 0.0, "steer": 0.0, "gnss": None, "scan": None}
    for _ in range(int(duration / cfg.dt)):
        enu = None if meas["gnss"] is None else sim.geo.to_enu(*meas["gnss"])
        cmd = stack.step(cfg.dt, meas["v"], meas["steer"], enu, meas["scan"])
        meas = sim.step(cmd.accel, cmd.steer)
        est = stack.loc.pose
        log.append((sim.t, *sim.car.pose, *(est if est is not None else (np.nan,) * 3),
                    *(enu[:2] if enu is not None else (np.nan, np.nan))))
    log = np.array(log)
    ok = ~np.isnan(log[:, 4])
    err = np.linalg.norm(log[ok, 1:3] - log[ok, 4:6], axis=1)
    yaw_err = np.abs(wrap_angle(log[ok, 3] - log[ok, 6]))
    gps = ~np.isnan(log[:, 7])
    antenna = np.array([transform_points(np.array([cfg.lever_arm]), p)[0] for p in log[gps, 1:4]])
    gps_err = np.linalg.norm(antenna - log[gps, 7:9], axis=1)
    summary = {
        "pos_rmse": float(np.sqrt(np.mean(err ** 2))),
        "pos_max": float(err.max()),
        "yaw_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(yaw_err ** 2)))),
        "gnss_raw_rmse": float(np.sqrt(np.mean(gps_err ** 2))),
        **stack.loc.stats,
    }
    if verbose:
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))
    return log, summary, sim, stack


def plot(log, sim, stack, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for x0, y0, x1, y1 in sim.segs:
        ax.plot([x0, x1], [y0, y1], "k-", lw=1)
    ax.plot(*stack.path.xy.T, "--", color="0.6", label="reference path")
    ax.plot(log[:, 7], log[:, 8], ".", ms=2, color="tab:orange", alpha=0.5, label="GNSS fixes")
    ax.plot(log[:, 1], log[:, 2], "-", color="tab:green", lw=2, label="ground truth")
    ax.plot(log[:, 4], log[:, 5], "-", color="tab:blue", lw=1, label="EKF (GNSS + LiDAR)")
    loc = stack.loc
    if not loc.mapper.empty and loc.lidar_pose is not None:
        # lidar-odom frame -> ENU using the current estimate: T = pose (+) lidar_pose^-1
        T = compose(loc.pose, ominus(loc.lidar_pose, np.zeros(3)))
        ax.plot(*transform_points(loc.mapper.points, T).T, ".", ms=1, color="tab:red", alpha=0.5,
                label="LiDAR local map")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("simple_robot - standalone 2D simulation")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gnss-std", type=float, default=0.5)
    ap.add_argument("--v-max", type=float, default=6.0)
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--no-gnss", action="store_true", help="GNSS only used for initialisation")
    ap.add_argument("--gnss-outage", type=float, nargs=2, metavar=("START", "END"),
                    help="simulate a GNSS outage between these times [s]")
    ap.add_argument("--plot", default=None, help="save a PNG of the run")
    a = ap.parse_args()
    log, _, sim, stack = run(a.duration, a.seed, not a.no_lidar, not a.no_gnss, a.gnss_std, a.v_max, a.gnss_outage)
    if a.plot:
        plot(log, sim, stack, a.plot)
