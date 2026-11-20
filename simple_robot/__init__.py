"""simple_robot: a minimal Ackermann-steered robot with GPS + LiDAR localization.

The core stack (vehicle model, control, planning, localization) is pure
numpy/scipy and does not depend on CARLA.  The CARLA bridge lives in
:mod:`simple_robot.carla_bridge` and is only imported when you run in CARLA.
"""

__version__ = "0.1.0"
