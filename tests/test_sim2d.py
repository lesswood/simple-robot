import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from run_sim2d import run  # noqa: E402


def test_full_stack_follows_loop_and_localizes():
    log, s, _, stack = run(duration=60.0, verbose=False)
    assert stack.loc.initialized
    assert s["pos_rmse"] < 0.4 and s["pos_max"] < 1.0
    assert s["yaw_rmse_deg"] < 2.5
    assert s["lidar_ok"] > 0.95 * (s["lidar_ok"] + s["lidar_rej"])
    # robot stayed on the road: max distance from reference path
    import numpy as np
    d = np.min(np.linalg.norm(log[:, None, 1:3] - stack.path.xy[None], axis=2), axis=1)
    assert d[log[:, 0] > 10].max() < 1.5


def test_lidar_bridges_gnss_outage():
    _, with_lidar, _, _ = run(duration=80.0, gnss_outage=(30, 60), verbose=False)
    _, without, _, _ = run(duration=80.0, gnss_outage=(30, 60), use_lidar=False, verbose=False)
    assert with_lidar["pos_max"] < without["pos_max"]
