"""CARLA bridge: spawns the robot and its sensors, converts data to the stack's conventions.

Coordinate conventions
----------------------
CARLA (Unreal Engine) is *left-handed*: x forward/east, y right/south, z up,
yaw in degrees clockwise.  The stack uses a right-handed ENU frame: x east,
y north, yaw counter-clockwise in radians.  Conversion: ``(x, y) -> (x, -y)``,
``yaw -> -radians(yaw)``.  CARLA's GNSS uses a Mercator projection about the
map geo-reference such that east = x and north = -y exactly, so with
:class:`~simple_robot.gnss.CarlaGeoProjection` GNSS fixes land in the same ENU
frame as the (converted) CARLA world coordinates; ground truth is used only
for evaluation, never by the stack.

The robot's body frame origin is the centre of the rear axle (the reference
point of the kinematic bicycle model).

Requires CARLA >= 0.9.14 for ``--control ackermann`` (``VehicleAckermannControl``);
``--control pid`` works with any 0.9.x release.
"""
from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np

from .gnss import CarlaGeoProjection
from .lidar import cloud_to_scan, voxel_downsample
from .vehicle import AckermannParams

# ---------------------------------------------------------------------------
# Pure conversion helpers (no carla import, unit-tested)
# ---------------------------------------------------------------------------


def carla_xy_to_enu(x: float, y: float) -> np.ndarray:
    return np.array([x, -y])


def enu_to_carla_xy(e: float, n: float) -> tuple[float, float]:
    return float(e), float(-n)


def carla_yaw_to_enu(yaw_deg: float) -> float:
    return float(np.arctan2(np.sin(-np.deg2rad(yaw_deg)), np.cos(-np.deg2rad(yaw_deg))))


def carla_lidar_to_body(raw: np.ndarray, mount_body_xyz) -> np.ndarray:
    """CARLA lidar points (N x 4: x, y, z, intensity in sensor frame) -> N x 3 right-handed body frame."""
    pts = np.array(raw[:, :3], dtype=float)
    pts[:, 1] *= -1.0
    return pts + np.asarray(mount_body_xyz, dtype=float)


def remove_own_body(points_xyz: np.ndarray, params: AckermannParams, pad: float = 0.3) -> np.ndarray:
    """Drop LiDAR returns from the robot's own body (body frame, rear-axle origin)."""
    x, y = points_xyz[:, 0], points_xyz[:, 1]
    inside = ((x > -params.rear_overhang - pad) & (x < params.length - params.rear_overhang + pad)
              & (np.abs(y) < params.width / 2.0 + pad))
    return points_xyz[~inside]


def metres_to_lat_lon_deg(sigma_m: float, lat0: float) -> tuple[float, float]:
    """Approximate GNSS noise standard deviation in degrees (CARLA's noise attributes are in degrees)."""
    m_per_deg = 111_320.0
    return sigma_m / m_per_deg, sigma_m / (m_per_deg * max(np.cos(np.deg2rad(lat0)), 1e-6))


# ---------------------------------------------------------------------------
# CARLA robot
# ---------------------------------------------------------------------------


@dataclass
class CarlaConfig:
    host: str = "localhost"
    port: int = 2000
    town: str | None = None             # e.g. "Town03"; None keeps the current map
    blueprint: str = "vehicle.tesla.model3"
    spawn_index: int = 0
    dt: float = 0.05                    # synchronous fixed time step [s]
    control: str = "ackermann"          # "ackermann" (CARLA >= 0.9.14) or "pid"
    # GNSS
    gnss_mount: tuple = (0.0, 0.0, 1.6)  # relative to the CARLA actor origin (CARLA axes)
    gnss_std: float = 0.5                # [m] noise added by CARLA
    gnss_rate: float = 5.0               # [Hz] (decimated in software)
    # LiDAR
    lidar_mount: tuple = (0.0, 0.0, 2.2)
    lidar_channels: int = 32
    lidar_range: float = 50.0
    lidar_pps: int = 200_000
    lidar_upper_fov: float = 10.0
    lidar_lower_fov: float = -25.0
    lidar_rate: float = 10.0             # [Hz] processing rate (decimated in software)
    scan_z_min: float = 0.5              # height band (above ground) flattened into the 2D scan
    scan_z_max: float = 2.5
    obstacle_z_min: float = 0.3          # height band used for obstacle avoidance (lower: catches kerb-high
    obstacle_z_max: float = 2.0          # objects, children, bollards; upper: anything the body could hit)
    # Vehicle dynamics assumed by the planner (braking is friction limited, so independent of mass)
    max_decel: float = 5.0               # [m/s^2]
    accel_tau: float = 0.3               # [s] drivetrain/brake response lag
    # Wheel odometry noise (emulates encoders + steering angle sensor)
    speed_noise: float = 0.05
    steer_noise_deg: float = 0.3
    seed: int = 0


class CarlaRobot:
    """Owns the CARLA connection, the vehicle and its sensors for one run."""

    def __init__(self, cfg: CarlaConfig):
        import carla  # imported lazily so the rest of the package works without CARLA

        self.carla = carla
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.client = carla.Client(cfg.host, cfg.port)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world(cfg.town) if cfg.town else self.client.get_world()
        self.map = self.world.get_map()
        self._orig_settings = self.world.get_settings()
        self.actors = []
        self.frame = None
        self.tick_count = 0

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = cfg.dt
        self.world.apply_settings(settings)

        geo = self.map.transform_to_geolocation(carla.Location(0.0, 0.0, 0.0))
        self.geo = CarlaGeoProjection(geo.latitude, geo.longitude, geo.altitude)
        self._spawn_vehicle()
        self._spawn_sensors()
        self.world.tick()  # let everything settle into place

    # ------------------------------------------------------------- spawning
    def _spawn_vehicle(self) -> None:
        carla, cfg = self.carla, self.cfg
        lib = self.world.get_blueprint_library()
        matches = lib.filter(cfg.blueprint)
        bp = matches[0] if matches else lib.filter("vehicle.tesla.model3")[0]
        bp.set_attribute("role_name", "hero")
        spawns = self.map.get_spawn_points()
        self.spawn_tf = spawns[cfg.spawn_index % len(spawns)]
        self.vehicle = self.world.spawn_actor(bp, self.spawn_tf)
        self.actors.append(self.vehicle)
        self.world.tick()

        # Geometry from the physics model: wheel positions are world coordinates in cm.
        phys = self.vehicle.get_physics_control()
        w = [np.array([wh.position.x, wh.position.y, wh.position.z]) / 100.0 for wh in phys.wheels]
        front, rear = (w[0] + w[1]) / 2.0, (w[2] + w[3]) / 2.0
        tf = self.vehicle.get_transform()
        fwd = tf.get_forward_vector()
        fwd = np.array([fwd.x, fwd.y, fwd.z])
        loc = np.array([tf.location.x, tf.location.y, tf.location.z])
        self.rear_axle_offset = float(np.dot(rear - loc, fwd))  # actor origin -> rear axle, along x
        wheelbase = float(np.linalg.norm(front - rear))
        track = float(np.linalg.norm(w[0] - w[1]))
        max_steer = np.deg2rad(phys.wheels[0].max_steer_angle)
        # Footprint from the bounding box (actor frame, centre + half extents)
        bb = self.vehicle.bounding_box
        rear_bumper = bb.location.x - bb.extent.x
        self.params = AckermannParams(
            wheelbase=wheelbase, track=track, max_steer=max_steer,
            length=2.0 * bb.extent.x, width=2.0 * bb.extent.y,
            rear_overhang=self.rear_axle_offset - rear_bumper,
            max_decel=cfg.max_decel, accel_tau=cfg.accel_tau)

    def _spawn_sensors(self) -> None:
        carla, cfg = self.carla, self.cfg
        lib = self.world.get_blueprint_library()

        gbp = lib.find("sensor.other.gnss")
        slat, slon = metres_to_lat_lon_deg(cfg.gnss_std, self.geo.lat0)
        for k, v in (("noise_lat_stddev", slat), ("noise_lon_stddev", slon), ("noise_seed", cfg.seed)):
            if gbp.has_attribute(k):
                gbp.set_attribute(k, str(v))
        gx, gy, gz = cfg.gnss_mount
        self.gnss = self.world.spawn_actor(gbp, carla.Transform(carla.Location(gx, gy, gz)), attach_to=self.vehicle)

        lbp = lib.find("sensor.lidar.ray_cast")
        attrs = {
            "channels": cfg.lidar_channels, "range": cfg.lidar_range, "points_per_second": cfg.lidar_pps,
            "rotation_frequency": 1.0 / cfg.dt,  # one full revolution per simulation tick
            "upper_fov": cfg.lidar_upper_fov, "lower_fov": cfg.lidar_lower_fov,
            "dropoff_general_rate": 0.0, "dropoff_zero_intensity": 0.0, "noise_stddev": 0.01,
        }
        for k, v in attrs.items():
            if lbp.has_attribute(k):
                lbp.set_attribute(k, str(v))
        lx, ly, lz = cfg.lidar_mount
        self.lidar = self.world.spawn_actor(lbp, carla.Transform(carla.Location(lx, ly, lz)), attach_to=self.vehicle)
        cbp = lib.find("sensor.other.collision")
        self.collision = self.world.spawn_actor(cbp, carla.Transform(), attach_to=self.vehicle)
        self.collisions: list = []      # (frame, other actor type) - evaluation only
        self.collision.listen(lambda e: self.collisions.append((e.frame, getattr(e.other_actor, "type_id", "?"))))
        self.actors += [self.gnss, self.lidar, self.collision]
        self.walkers: list = []         # (walker actor, crossing point (CARLA xy), direction (CARLA xy), half-width)

        self._q_gnss: queue.Queue = queue.Queue()
        self._q_lidar: queue.Queue = queue.Queue()
        self.gnss.listen(self._q_gnss.put)
        self.lidar.listen(self._q_lidar.put)

        # Mount positions in the right-handed rear-axle body frame
        self.gnss_lever_arm = (gx - self.rear_axle_offset, -gy)
        self.lidar_mount_body = (lx - self.rear_axle_offset, -ly, lz)

    # ---------------------------------------------------------------- tick
    def _get(self, q: queue.Queue, frame: int, timeout: float = 5.0):
        while True:
            data = q.get(timeout=timeout)
            if data.frame == frame:
                return data
            if data.frame > frame:  # should not happen in synchronous mode
                return None

    def tick(self) -> dict:
        """Advance the simulation one step and return the measurements of that step."""
        cfg = self.cfg
        self.frame = self.world.tick()
        self.tick_count += 1
        gnss = self._get(self._q_gnss, self.frame)
        lidar = self._get(self._q_lidar, self.frame)

        self._update_walkers()
        meas = {"v": self._wheel_speed(), "steer": self._steer_angle(), "gnss": None, "scan": None, "obstacles": None}
        if gnss is not None and self.tick_count % max(1, round(1.0 / (cfg.gnss_rate * cfg.dt))) == 0:
            meas["gnss"] = self.geo.to_enu(gnss.latitude, gnss.longitude, gnss.altitude)
        if lidar is not None and self.tick_count % max(1, round(1.0 / (cfg.lidar_rate * cfg.dt))) == 0:
            raw = np.frombuffer(lidar.raw_data, dtype=np.float32).reshape(-1, 4)
            pts = remove_own_body(carla_lidar_to_body(raw, self.lidar_mount_body), self.params)
            meas["scan"] = cloud_to_scan(pts, cfg.scan_z_min, cfg.scan_z_max, r_min=0.0, r_max=cfg.lidar_range)
            obs = cloud_to_scan(pts, cfg.obstacle_z_min, cfg.obstacle_z_max, r_min=0.0, r_max=40.0)
            meas["obstacles"] = voxel_downsample(obs, 0.15)
        return meas

    def _wheel_speed(self) -> float:
        vel = self.vehicle.get_velocity()
        fwd = self.vehicle.get_transform().get_forward_vector()
        v = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z
        return float(v + self.rng.normal(0.0, self.cfg.speed_noise)) if abs(v) > 0.05 else 0.0

    def _steer_angle(self) -> float:
        # CARLA steer is normalised [-1, 1] with positive = right; ours is radians, positive = left.
        steer = -self.vehicle.get_control().steer * self.params.max_steer
        return float(steer + self.rng.normal(0.0, np.deg2rad(self.cfg.steer_noise_deg)))

    # -------------------------------------------------------------- control
    def apply(self, speed: float, accel: float, steer: float, stop: bool = False, reverse: bool = False) -> None:
        """Apply a command from the stack: signed target speed [m/s] (negative = reverse), signed
        acceleration request [m/s^2] and steering angle [rad, + = left]."""
        carla = self.carla
        steer = float(np.clip(steer, -self.params.max_steer, self.params.max_steer))
        if self.cfg.control == "ackermann":
            # VehicleAckermannControl: steer [rad, + = right], speed [m/s, < 0 reverses],
            # acceleration = magnitude used to reach that speed (we pass the braking demand through).
            mag = float(np.clip(abs(accel), 0.5, self.params.max_decel))
            ctrl = carla.VehicleAckermannControl(
                steer=-steer, steer_speed=np.deg2rad(120.0),
                speed=0.0 if stop else float(speed), acceleration=self.params.max_decel if stop else mag, jerk=0.0)
            self.vehicle.apply_ackermann_control(ctrl)
        else:
            drive = -accel if reverse else accel          # positive = push in the direction of travel
            throttle = 0.0 if stop else float(np.clip(drive / 3.0, 0.0, 0.75))
            brake = 1.0 if stop else float(np.clip(-drive / self.params.max_decel, 0.0, 1.0))
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=throttle, brake=brake, steer=float(-steer / self.params.max_steer), reverse=bool(reverse)))

    # ------------------------------------------------------------- scenario
    def drivable(self, xy: np.ndarray) -> np.ndarray:
        """True for ENU points on a driving lane (used to keep swerves on the road)."""
        carla = self.carla
        out = np.zeros(len(xy), dtype=bool)
        for i, (e, n) in enumerate(xy):
            x, y = enu_to_carla_xy(e, n)
            wp = self.map.get_waypoint(carla.Location(x, y, 0.5), project_to_road=False,
                                       lane_type=carla.LaneType.Driving)
            out[i] = wp is not None
        return out

    def _route_frame(self, route_xy: np.ndarray, s: float):
        """Point (ENU) and heading of the route at arc length ``s``."""
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(route_xy, axis=0), axis=1))])
        i = int(np.clip(np.searchsorted(d, s), 1, len(route_xy) - 1))
        seg = route_xy[i] - route_xy[i - 1]
        return route_xy[i], float(np.arctan2(seg[1], seg[0]))

    def spawn_parked_cars(self, route_xy: np.ndarray, n: int, start: float = 35.0, spacing: float = 45.0,
                          lateral: float = 1.0) -> int:
        """Park ``n`` static cars along the route, alternating sides, partly blocking the lane."""
        carla = self.carla
        lib = self.world.get_blueprint_library()
        bps = [b for b in lib.filter("vehicle.*") if not b.has_attribute("number_of_wheels")
               or int(b.get_attribute("number_of_wheels").as_int()) == 4] or lib.filter("vehicle.*")
        spawned = 0
        for i in range(n):
            (e, nn), yaw = self._route_frame(route_xy, start + i * spacing)
            side = 1.0 if i % 2 == 0 else -1.0                         # + = left of the route
            e, nn = e - side * lateral * np.sin(yaw), nn + side * lateral * np.cos(yaw)
            x, y = enu_to_carla_xy(e, nn)
            bp = bps[int(self.rng.integers(len(bps)))]
            tf = carla.Transform(carla.Location(x, y, 0.3), carla.Rotation(yaw=-np.rad2deg(yaw)))
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is not None:
                actor.set_simulate_physics(False)
                self.actors.append(actor)
                spawned += 1
        return spawned

    def spawn_walkers(self, route_xy: np.ndarray, n: int, start: float = 60.0, spacing: float = 50.0,
                      half_width: float = 6.0, speed: float = 1.2) -> int:
        """Pedestrians pacing back and forth across the route (driven directly, no AI navigation)."""
        carla = self.carla
        bps = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        spawned = 0
        for i in range(n):
            (e, nn), yaw = self._route_frame(route_xy, start + i * spacing)
            normal = np.array([-np.sin(yaw), np.cos(yaw)])
            p0 = np.array([e, nn]) + normal * half_width * (1 if i % 2 == 0 else -1)
            x, y = enu_to_carla_xy(*p0)
            bp = bps[int(self.rng.integers(len(bps)))]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker = self.world.try_spawn_actor(bp, carla.Transform(carla.Location(x, y, 1.0)))
            if walker is None:
                continue
            self.actors.append(walker)
            centre = np.array(enu_to_carla_xy(e, nn))
            direction = centre - np.array([x, y])
            self.walkers.append([walker, centre, direction / np.linalg.norm(direction), half_width, speed])
            spawned += 1
        return spawned

    def _update_walkers(self) -> None:
        carla = self.carla
        for w in self.walkers:
            walker, centre, direction, half, speed = w
            loc = walker.get_transform().location
            along = float(np.dot(np.array([loc.x, loc.y]) - centre, direction))
            if along > half:            # reached the far side: turn around
                w[2] = direction = -direction
            walker.apply_control(carla.WalkerControl(
                direction=carla.Vector3D(float(direction[0]), float(direction[1]), 0.0), speed=speed))

    # ------------------------------------------------------------- helpers
    def ground_truth(self) -> np.ndarray:
        """True rear-axle pose in ENU (evaluation only)."""
        tf = self.vehicle.get_transform()
        yaw = carla_yaw_to_enu(tf.rotation.yaw)
        xy = carla_xy_to_enu(tf.location.x, tf.location.y)
        xy = xy + self.rear_axle_offset * np.array([np.cos(yaw), np.sin(yaw)])
        return np.array([xy[0], xy[1], yaw])

    def route(self, length: float = 300.0, step: float = 1.0, seed: int = 0) -> np.ndarray:
        """Follow the lane graph from the spawn point; returns ENU points (Nx2)."""
        rng = np.random.default_rng(seed)
        wp = self.map.get_waypoint(self.spawn_tf.location)
        pts = [carla_xy_to_enu(wp.transform.location.x, wp.transform.location.y)]
        travelled = 0.0
        while travelled < length:
            nxt = wp.next(step)
            if not nxt:
                break
            wp = nxt[int(rng.integers(len(nxt)))]
            pts.append(carla_xy_to_enu(wp.transform.location.x, wp.transform.location.y))
            travelled += step
        return np.array(pts)

    def draw_path(self, xy: np.ndarray, life_time: float = 120.0) -> None:
        carla = self.carla
        for e, n in xy[::2]:
            x, y = enu_to_carla_xy(e, n)
            self.world.debug.draw_point(carla.Location(x, y, 0.3), size=0.06,
                                        color=carla.Color(0, 160, 255), life_time=life_time)

    def draw_estimate(self, pose, life_time: float = 60.0) -> None:
        carla = self.carla
        x, y = enu_to_carla_xy(pose[0], pose[1])
        self.world.debug.draw_point(carla.Location(x, y, 0.5), size=0.08,
                                    color=carla.Color(255, 40, 40), life_time=life_time)

    def draw_local_plan(self, xy_odom: np.ndarray, map_in_odom, life_time: float = 0.6) -> None:
        """Draw the local plan (given in the stack's odom frame)."""
        from .geometry import ominus, transform_points
        carla = self.carla
        xy = transform_points(np.asarray(xy_odom)[:40:3], ominus(map_in_odom, np.zeros(3)))
        for e, n in xy:
            x, y = enu_to_carla_xy(e, n)
            self.world.debug.draw_point(carla.Location(x, y, 0.4), size=0.05,
                                        color=carla.Color(40, 220, 40), life_time=life_time)

    def follow_with_spectator(self) -> None:
        carla = self.carla
        tf = self.vehicle.get_transform()
        fwd = tf.get_forward_vector()
        loc = tf.location - carla.Location(fwd.x * 10.0, fwd.y * 10.0, 0.0) + carla.Location(z=6.0)
        self.world.get_spectator().set_transform(
            carla.Transform(loc, carla.Rotation(pitch=-25.0, yaw=tf.rotation.yaw)))

    def destroy(self) -> None:
        for s in (getattr(self, "gnss", None), getattr(self, "lidar", None), getattr(self, "collision", None)):
            if s is not None and s.is_alive:
                s.stop()
        self.client.apply_batch([self.carla.command.DestroyActor(a) for a in self.actors if a.is_alive])
        self.world.apply_settings(self._orig_settings)
