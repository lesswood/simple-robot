"""A minimal stand-in for the ``carla`` Python API, used to test the bridge without a server.

It implements just enough of the API that :mod:`simple_robot.carla_bridge`
touches, with a kinematic vehicle, a GNSS sensor using LibCarla's geo
projection and a lidar that sees a ring of walls.  CARLA conventions
(left-handed, degrees, steer positive = right, wheel positions in cm) are
reproduced so sign errors in the bridge show up in tests.
"""
from __future__ import annotations

import math
import types

import numpy as np

R_EARTH = 6378137.0
LAT0, LON0 = 49.0, 8.0


class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o):
        return Location(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return Location(self.x - o.x, self.y - o.y, self.z - o.z)


Vector3D = Location


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()

    def get_forward_vector(self):
        y = math.radians(self.rotation.yaw)
        return Vector3D(math.cos(y), math.sin(y), 0.0)


class GeoLocation:
    def __init__(self, lat, lon, alt):
        self.latitude, self.longitude, self.altitude = lat, lon, alt


def _geo(loc):
    scale = math.cos(math.radians(LAT0))
    mx = scale * math.radians(LON0) * R_EARTH + loc.x
    my = scale * R_EARTH * math.log(math.tan((90 + LAT0) * math.pi / 360)) - loc.y
    lon = mx * 180 / (math.pi * R_EARTH * scale)
    lat = 360 * math.atan(math.exp(my / (R_EARTH * scale))) / math.pi - 90
    return GeoLocation(lat, lon, loc.z)


class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, reverse=False):
        self.throttle, self.steer, self.brake, self.reverse = throttle, steer, brake, reverse


class VehicleAckermannControl:
    def __init__(self, steer=0.0, steer_speed=0.0, speed=0.0, acceleration=0.0, jerk=0.0):
        self.steer, self.speed, self.acceleration = steer, speed, acceleration


class WalkerControl:
    def __init__(self, direction=None, speed=0.0, jump=False):
        self.direction, self.speed = direction or Vector3D(), speed


LaneType = types.SimpleNamespace(Driving=1, Any=-1)


class Color:
    def __init__(self, r=0, g=0, b=0):
        pass


class _Attr:
    def __init__(self, bp_id):
        self.id = bp_id
        self.attrs = {}

    def has_attribute(self, k):
        return k != "number_of_wheels"

    def set_attribute(self, k, v):
        self.attrs[k] = v


class _Lib:
    def filter(self, pattern):
        return [_Attr(pattern)] if pattern.startswith(("vehicle.", "walker.")) else []

    def find(self, bp_id):
        return _Attr(bp_id)


class _Wheel:
    def __init__(self, x, y):
        self.position = Vector3D(x * 100, y * 100, 0.0)
        self.max_steer_angle = 70.0


class _Data:
    def __init__(self, frame, **kw):
        self.frame = frame
        self.__dict__.update(kw)


class Actor:
    _ids = 0

    def __init__(self, world, bp, tf, parent=None):
        Actor._ids += 1
        self.id = Actor._ids
        self.world, self.bp, self.tf, self.parent = world, bp, tf, parent
        self.is_alive = True
        self.cb = None

    def listen(self, cb):
        self.cb = cb

    def set_simulate_physics(self, enabled):
        pass

    @property
    def type_id(self):
        return self.bp.id

    def footprint(self, n=12):
        """Outline points (CARLA xy) used by the fake lidar / collision check; None for sensors."""
        size = {"vehicle": (4.6, 1.9), "walker": (0.5, 0.5)}.get(self.bp.id.split(".")[0])
        if size is None:
            return None
        L, W = size
        u = np.linspace(-0.5, 0.5, n)
        pts = np.vstack([np.column_stack([u * L, np.full(n, -W / 2)]), np.column_stack([u * L, np.full(n, W / 2)]),
                         np.column_stack([np.full(n, -L / 2), u * W]), np.column_stack([np.full(n, L / 2), u * W])])
        y = math.radians(self.tf.rotation.yaw)
        R = np.array([[math.cos(y), -math.sin(y)], [math.sin(y), math.cos(y)]])
        return pts @ R.T + [self.tf.location.x, self.tf.location.y]

    def stop(self):
        self.cb = None

    def get_transform(self):
        return self.tf


class Walker(Actor):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ctrl = WalkerControl()

    def apply_control(self, c):
        self.ctrl = c

    def physics(self, dt):
        d = self.ctrl.direction
        self.tf = Transform(Location(self.tf.location.x + d.x * self.ctrl.speed * dt,
                                     self.tf.location.y + d.y * self.ctrl.speed * dt, self.tf.location.z))


class Vehicle(Actor):
    L = 2.8

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.v = 0.0
        self.steer_rad = 0.0     # CARLA convention: positive = right
        self.ctrl = VehicleControl()
        # actor origin at the bounding-box centre; rear axle 1.4 m behind it
        self.bounding_box = types.SimpleNamespace(extent=Vector3D(2.3, 0.95, 0.8), location=Location(0.0, 0.0, 0.8))

    def get_physics_control(self):
        # rear axle 1.4 m behind the actor origin, front 1.4 m ahead (world coords, cm)
        t = self.tf
        f = t.get_forward_vector()
        r = Vector3D(-f.y, f.x, 0)  # right vector in left-handed frame
        pts = [(1.4, -0.8), (1.4, 0.8), (-1.4, -0.8), (-1.4, 0.8)]
        w = [_Wheel(t.location.x + a * f.x + b * r.x, t.location.y + a * f.y + b * r.y) for a, b in pts]
        return types.SimpleNamespace(wheels=w)

    def get_velocity(self):
        f = self.tf.get_forward_vector()
        return Vector3D(f.x * self.v, f.y * self.v, 0)

    def get_control(self):
        return VehicleControl(steer=self.steer_rad / math.radians(70.0))

    def apply_ackermann_control(self, c):
        self.target_speed, self.steer_rad = c.speed, c.steer

    def apply_control(self, c):
        sign = -1.0 if getattr(c, "reverse", False) else 1.0
        push = c.throttle * 3.0 * sign
        brake = -np.sign(self.v) * c.brake * 6.0
        self.target_speed = self.v + push + brake
        self.steer_rad = c.steer * math.radians(70.0)

    def physics(self, dt):
        if not hasattr(self, "target_speed"):
            return                        # parked
        dv = np.clip(self.target_speed - self.v, -4 * dt, 2 * dt)
        self.v = self.v + dv
        # integrate rear axle (left-handed: positive steer -> yaw increases, i.e. turning right)
        yaw = math.radians(self.tf.rotation.yaw)
        rear = (self.tf.location.x - 1.4 * math.cos(yaw), self.tf.location.y - 1.4 * math.sin(yaw))
        rear = (rear[0] + self.v * dt * math.cos(yaw), rear[1] + self.v * dt * math.sin(yaw))
        yaw += self.v * math.tan(self.steer_rad) / self.L * dt
        self.tf = Transform(Location(rear[0] + 1.4 * math.cos(yaw), rear[1] + 1.4 * math.sin(yaw), 0.0),
                            Rotation(yaw=math.degrees(yaw)))


class Waypoint:
    def __init__(self, x, y):
        self.transform = Transform(Location(x, y, 0))

    def next(self, d):
        # a straight lane along +x, then curving: keep it simple (straight)
        return [Waypoint(self.transform.location.x + d, self.transform.location.y)]


class Map:
    def transform_to_geolocation(self, loc):
        return _geo(loc)

    def get_spawn_points(self):
        return [Transform(Location(0, 0, 0.5), Rotation(yaw=0.0))]

    def get_waypoint(self, loc, project_to_road=True, lane_type=None):
        if not project_to_road and abs(loc.y) > 7.5:    # off the road (walls at y = +-9)
            return None
        return Waypoint(loc.x, loc.y)


class World:
    def __init__(self):
        self.frame = 0
        self.actors = []
        self.settings = types.SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None)
        self.debug = types.SimpleNamespace(draw_point=lambda *a, **k: None)
        self._spectator = types.SimpleNamespace(set_transform=lambda tf: None)
        rng = np.random.default_rng(0)
        # walls along both sides of the road plus posts
        xs = np.arange(-20, 400, 0.25)
        self.walls = np.vstack([np.column_stack([xs, np.full_like(xs, -9)]),
                                np.column_stack([xs, np.full_like(xs, 9)]),
                                *[np.column_stack([np.full(20, x), np.linspace(-9, -6, 20)])
                                  for x in rng.uniform(0, 400, 60)],
                                *[np.column_stack([np.full(20, x), np.linspace(6, 9, 20)])
                                  for x in rng.uniform(0, 400, 60)]])

    def get_map(self):
        return Map()

    def get_settings(self):
        return types.SimpleNamespace(**vars(self.settings))

    def apply_settings(self, s):
        self.settings = s

    def get_blueprint_library(self):
        return _Lib()

    def get_spectator(self):
        return self._spectator

    def spawn_actor(self, bp, tf, attach_to=None):
        cls = Vehicle if bp.id.startswith("vehicle.") else Walker if bp.id.startswith("walker.") else Actor
        a = cls(self, bp, tf, attach_to) if cls is Actor else cls(self, bp, tf)
        self.actors.append(a)
        return a

    try_spawn_actor = spawn_actor

    def _obstacle_points(self, exclude):
        pts = [a.footprint() for a in self.actors if a is not exclude and a.footprint() is not None]
        return np.vstack([self.walls] + pts) if pts else self.walls

    def _sensor_world_tf(self, s):
        p = s.parent.tf
        yaw = math.radians(p.rotation.yaw)
        x = p.location.x + s.tf.location.x * math.cos(yaw) - s.tf.location.y * math.sin(yaw)
        y = p.location.y + s.tf.location.x * math.sin(yaw) + s.tf.location.y * math.cos(yaw)
        return Location(x, y, s.tf.location.z), yaw

    def tick(self):
        dt = self.settings.fixed_delta_seconds or 0.05
        for a in self.actors:
            if isinstance(a, (Vehicle, Walker)):
                a.physics(dt)
        self.frame += 1
        for s in self.actors:
            if s.cb is None:
                continue
            loc, yaw = self._sensor_world_tf(s)
            if s.bp.id == "sensor.other.gnss":
                g = _geo(loc)
                s.cb(_Data(self.frame, latitude=g.latitude, longitude=g.longitude, altitude=g.altitude))
            elif s.bp.id == "sensor.other.collision":
                hero = s.parent.footprint(n=30)
                for other in self.actors:
                    fp = other.footprint() if other is not s.parent else None
                    if fp is not None and _overlap(hero, fp):
                        s.cb(_Data(self.frame, other_actor=other))
            else:
                world_pts = self._obstacle_points(s.parent)
                rel = world_pts - [loc.x, loc.y]
                rel = rel[np.linalg.norm(rel, axis=1) < 40]
                c, sn = math.cos(yaw), math.sin(yaw)
                sx = rel[:, 0] * c + rel[:, 1] * sn       # world -> sensor (left-handed)
                sy = -rel[:, 0] * sn + rel[:, 1] * c
                z = np.full_like(sx, 1.0 - loc.z)          # 1 m above ground
                raw = np.column_stack([sx, sy, z, np.ones_like(sx)]).astype(np.float32)
                s.cb(_Data(self.frame, raw_data=raw.tobytes()))
        return self.frame


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Crude polygon overlap from outline points: any point of one inside the other's bounding box."""
    def inside(p, poly):
        c = poly.mean(0)
        # principal axes of the outline
        u, _, vt = np.linalg.svd(poly - c, full_matrices=False)
        q = (p - c) @ vt.T
        ext = np.abs((poly - c) @ vt.T).max(0)
        return np.any(np.all(np.abs(q) <= ext + 1e-6, axis=1))
    return inside(a, b) or inside(b, a)


class Client:
    def __init__(self, host, port):
        self.world = World()

    def set_timeout(self, t):
        pass

    def get_world(self):
        return self.world

    def load_world(self, town):
        return self.world

    def apply_batch(self, cmds):
        pass


command = types.SimpleNamespace(DestroyActor=lambda a: a)
