#!/usr/bin/env python3
"""Run the full stack in the standalone 2D simulator (no CARLA required).

    python scripts/run_sim2d.py --plot sim2d.png                 # obstacles + avoidance
    python scripts/run_sim2d.py --no-obstacles --plot clear.png  # localization only
    python scripts/run_sim2d.py --no-avoidance                   # see it crash
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
from simple_robot.sim2d import DEFAULT_MOVERS, DEFAULT_OBSTACLES, Sim2D, SimConfig  # noqa: E402
from simple_robot.vehicle import AckermannParams  # noqa: E402


def run(duration: float = 90.0, seed: int = 0, use_lidar: bool = True, use_gnss: bool = True,
        gnss_std: float = 0.5, v_max: float = 6.0, gnss_outage=None, obstacles: bool = False,
        avoidance: bool = True, extra_obstacles=(), verbose: bool = True):
    params = AckermannParams(wheelbase=2.8, track=1.6, accel_tau=0.3)
    cfg = SimConfig(gnss_std=gnss_std, gnss_outage=gnss_outage,
                    obstacles=tuple(DEFAULT_OBSTACLES if obstacles else ()) + tuple(extra_obstacles),
                    movers=tuple(DEFAULT_MOVERS if obstacles else ()))
    sim = Sim2D(params, cfg, seed=seed)
    path = Path(sim.path, closed=True)
    loc_cfg = LocalizerConfig(wheelbase=params.wheelbase, gnss_std=gnss_std, gnss_lever_arm=cfg.lever_arm,
                              use_lidar=use_lidar, use_gnss=use_gnss)
    stack = RobotStack(params, path, loc_cfg, v_max=v_max, avoidance=avoidance)

    log = []
    meas = {"v": 0.0, "steer": 0.0, "gnss": None, "scan": None}
    progress, blocked_time, reversing = 0.0, 0.0, 0.0
    gt_idx = path.nearest_index(sim.car.pose)
    for _ in range(int(duration / cfg.dt)):
        enu = None if meas["gnss"] is None else sim.geo.to_enu(*meas["gnss"])
        cmd = stack.step(cfg.dt, meas["v"], meas["steer"], enu, meas["scan"])
        meas = sim.step(cmd.accel, cmd.steer, cmd.reverse)
        new_idx = path.nearest_index(sim.car.pose, gt_idx)
        n = len(path.xy) - 1
        progress += ((new_idx - gt_idx + n // 2) % n - n // 2) * 0.5   # signed route progress [m]
        gt_idx = new_idx
        blocked_time += cfg.dt if cmd.blocked else 0.0
        reversing += cfg.dt if cmd.reverse else 0.0
        est = stack.loc.pose
        log.append((sim.t, *sim.car.pose, *(est if est is not None else (np.nan,) * 3),
                    *(enu[:2] if enu is not None else (np.nan, np.nan)), sim.car.v))
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
        "laps": float(progress / path.length),
        "collisions": sim.collisions,
        "hit_while_stopped": sim.collisions_stationary,
        "min_clearance": float(sim.min_clearance),
        "blocked_s": float(blocked_time),
        "reversing_s": float(reversing),
        **stack.loc.stats,
    }
    if verbose:
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))
    return log, summary, sim, stack


def plot(log, sim, stack, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for x0, y0, x1, y1 in sim.static_segs:
        ax.plot([x0, x1], [y0, y1], "k-", lw=1)
    for i, (cx, cy, w, h) in enumerate(sim.cfg.obstacles):
        ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, color="tab:red", alpha=0.6,
                                   label="obstacles" if i == 0 else None))
    for (x, y0, y1, _, size) in sim.cfg.movers:
        ax.add_patch(plt.Rectangle((x - size / 2, y0), size, y1 - y0, color="tab:purple", alpha=0.15))
        ax.text(x + 0.8, y1, "pedestrian", color="tab:purple", fontsize=7)
    ax.plot(*stack.path.xy.T, "--", color="0.6", label="reference route")
    ax.plot(log[:, 7], log[:, 8], ".", ms=2, color="tab:orange", alpha=0.5, label="GNSS fixes")
    sc = ax.scatter(log[:, 1], log[:, 2], c=log[:, 9], s=3, cmap="viridis", label="ground truth (colour = speed)")
    ax.plot(log[:, 4], log[:, 5], "-", color="tab:blue", lw=0.7, alpha=0.8, label="EKF estimate")
    loc = stack.loc
    if not loc.mapper.empty and loc.lidar_pose is not None:
        # lidar-odom frame -> ENU using the current estimate: T = pose (+) lidar_pose^-1
        T = compose(loc.pose, ominus(loc.lidar_pose, np.zeros(3)))
        ax.plot(*transform_points(loc.mapper.points, T).T, ".", ms=1, color="tab:red", alpha=0.5,
                label="LiDAR local map")
    fig.colorbar(sc, ax=ax, label="speed [m/s]", shrink=0.7)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("simple_robot: GNSS + LiDAR localization, speed-aware obstacle avoidance (2D sim)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gnss-std", type=float, default=0.5)
    ap.add_argument("--v-max", type=float, default=6.0)
    ap.add_argument("--no-lidar", action="store_true", help="LiDAR not used for localization")
    ap.add_argument("--no-gnss", action="store_true", help="GNSS only used for initialisation")
    ap.add_argument("--gnss-outage", type=float, nargs=2, metavar=("START", "END"),
                    help="simulate a GNSS outage between these times [s]")
    ap.add_argument("--no-obstacles", action="store_true", help="empty road (no parked car, crates, pedestrian)")
    ap.add_argument("--no-avoidance", action="store_true", help="follow the route blindly")
    ap.add_argument("--plot", default=None, help="save a PNG of the run")
    a = ap.parse_args()
    log, _, sim, stack = run(a.duration, a.seed, not a.no_lidar, not a.no_gnss, a.gnss_std, a.v_max,
                             a.gnss_outage, not a.no_obstacles, not a.no_avoidance)
    if a.plot:
        plot(log, sim, stack, a.plot)
