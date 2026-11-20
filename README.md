# simple-robot

A simple wheeled robot with **Ackermann steering** that localizes with **GNSS + LiDAR**
and drives a route in the **[CARLA](https://carla.org) simulator**.

![2D simulation: ground truth vs estimate](docs/sim2d.png)

*Standalone 2D simulation run (`scripts/run_sim2d.py --gnss-outage 40 70`): the EKF
estimate (blue) tracks ground truth (green) through a 30 s GNSS outage using LiDAR odometry.*

## Architecture

```
            CARLA (or the built-in 2D sim)
   ┌──────────────┬──────────────┬───────────────────┐
   │ GNSS lat/lon │ 3D LiDAR     │ wheel speed+steer │
   ▼              ▼              ▼                   │
 gnss.py       lidar.py         (odometry)           │
 lat/lon→ENU   height slice→2D                       │
   │           ICP vs local map                      │
   │           → relative motion                     │
   ▼              ▼              ▼                   │
 ┌──────────── localization.py / ekf.py ──────────┐  │
 │ EKF on (x, y, yaw), bicycle-model prediction   │  │
 │ GNSS: absolute update (with antenna lever arm) │  │
 │ LiDAR: relative update via stochastic cloning  │  │
 │ auto-init: heading from GNSS track vs odometry │  │
 └───────────────────────┬────────────────────────┘  │
                         ▼                           │
   planning.py (route, speed profile)                │
   control.py  (pure pursuit + PID)  ── robot.py ────┘
                         │ steer angle, speed
                         ▼
          carla_bridge.py → VehicleAckermannControl
```

| Module | What it does |
| --- | --- |
| `vehicle.py` | Ackermann geometry (inner/outer wheel angles), kinematic bicycle model about the rear axle |
| `gnss.py` | Exact inverse of CARLA's GNSS geo-projection, plus a WGS-84 ECEF→ENU converter for real hardware |
| `lidar.py` | 3D cloud → 2D scan, voxel filter, robust point-to-line ICP with Hessian covariance, sliding local map |
| `ekf.py` | SE(2) EKF with GNSS, absolute-pose and relative-pose (stochastic cloning) updates, χ² gating |
| `localization.py` | Initialization and sensor fusion logic |
| `planning.py`, `control.py` | Path with curvature-limited speed profile, pure-pursuit steering, PID speed |
| `robot.py` | The autonomy loop, independent of the simulator |
| `carla_bridge.py` | Spawns the vehicle and sensors in CARLA, runs synchronous mode, converts frames, applies control |
| `sim2d.py` | Lightweight 2D simulator (ray-cast LiDAR, noisy GNSS/odometry) for testing without CARLA |

### Localization design notes

* **GNSS** gives absolute position but no heading. At start-up the robot rolls forward
  slowly. A 2D rigid alignment between the dead-reckoned odometry track and the GNSS
  track then gives the initial position *and* heading.
* **LiDAR** is used as scan-to-local-map odometry. The local map is kept in its own
  "lidar-odom" frame and is never corrected by GNSS. The motion between consecutive
  scans is fused as a *relative* measurement through stochastic cloning, which keeps
  the LiDAR and GNSS information statistically independent. Inserting scans at the
  fused pose instead would make the LiDAR measurement echo the filter's own past
  estimate and override GNSS.
* ICP covariance comes from the point-to-line Hessian. In geometrically degenerate
  places, such as a featureless corridor, it automatically becomes uncertain along
  the corridor.
* Coordinate frames: CARLA is left-handed (y right, yaw clockwise in degrees). The
  stack is right-handed ENU (x east, y north, yaw counter-clockwise in radians).
  `carla_bridge.py` holds every conversion: `(x, y) → (x, −y)`, `yaw → −yaw`.

## Quick start (no CARLA needed)

```bash
pip install -r requirements.txt
python scripts/run_sim2d.py --plot sim.png                    # GNSS + LiDAR
python scripts/run_sim2d.py --gnss-outage 40 70 --plot o.png  # LiDAR bridges a GNSS outage
python scripts/run_sim2d.py --no-lidar                        # compare: GNSS + odometry only
python -m pytest
```

Typical results (120 s, 0.5 m GNSS noise, 3 % wheel-speed scale error):

| Configuration | Position RMSE | Max error | Yaw RMSE |
| --- | --- | --- | --- |
| GNSS + LiDAR | 0.22 m | 0.61 m | 1.1° |
| GNSS only | 0.32 m | 0.67 m | 1.5° |
| GNSS + LiDAR, 30 s GNSS outage | 0.62 m | 2.0 m | 1.5° |
| GNSS only, 30 s GNSS outage | 1.23 m | 4.9 m | 2.7° |

## Running in CARLA

1. Install CARLA **0.9.14 or newer** (needed for `VehicleAckermannControl`; use
   `--control pid` on older 0.9.x versions) and start the server:
   ```bash
   ./CarlaUE4.sh -quality-level=Low        # Windows: CarlaUE4.exe
   ```
2. Install the Python client matching the server version:
   ```bash
   pip install carla==0.9.15   # or the .whl from $CARLA_ROOT/PythonAPI/carla/dist
   ```
3. Run the robot:
   ```bash
   python scripts/run_carla.py --town Town03 --route-length 300 --plot carla.png
   ```

The robot spawns at spawn point `--spawn`, initializes its heading from GNSS, and then
follows a random route through the lane graph. In the CARLA window, blue dots are the
route and red dots the estimated pose. When the run ends, `carla_run.csv` holds ground
truth vs estimate, and the position and yaw RMSE are printed.

Useful options:

| Option | Default | Notes |
| --- | --- | --- |
| `--blueprint` | `vehicle.tesla.model3` | Any 4-wheeled vehicle; for a small robot-like car try `vehicle.micro.microlino` (CARLA ≥ 0.9.14). Wheelbase, track and max steer are read from the physics model |
| `--control` | `ackermann` | `pid` = throttle/brake/steer for older CARLA |
| `--gnss-std` | `0.5` | GNSS noise added by CARLA's sensor [m] |
| `--no-lidar` | off | GNSS + wheel odometry only |
| `--v-max` | `6.0` | Max speed [m/s] |

Sensor setup (`CarlaConfig` in `carla_bridge.py`): GNSS at 5 Hz on the roof; a 32-channel
LiDAR at 2.2 m running one revolution per 20 Hz simulation tick, processed at 10 Hz. A
0.5–2.5 m height band of each cloud is flattened into a 2D scan. Wheel speed and steering
angle come from the vehicle state with added noise, emulating encoders. Ground truth is
used only for evaluation.

### Testing the CARLA bridge without a server

`tests/fake_carla.py` is a small stand-in for the CARLA Python API that reproduces its
conventions: left-handed frame, steer positive to the right, wheel positions in cm, and
LibCarla's geo-projection. The test suite drives `scripts/run_carla.py` end-to-end against
it, which catches sign and frame errors in the bridge. It has not been run against a real
CARLA server here, so expect some tuning (scan height band, ICP thresholds) per map.

## Limitations and next steps

* The LiDAR map is local (a sliding ~40 m window), so with GNSS denied long enough the
  estimate drifts. For GNSS-free operation, add loop closure or localize against a prior
  map; `EKF.update_pose` is already there for absolute map-based fixes.
* Planar (2D) model: no pitch or roll, and slopes and ramps aren't handled.
* No obstacle avoidance or traffic: the route is followed blindly.
