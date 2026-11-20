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
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0):
        self.throttle, self.steer, self.brake = throttle, steer, brake


class VehicleAckermannControl:
    def __init__(self, steer=0.0, steer_speed=0.0, speed=0.0, acceleration=0.0, jerk=0.0):
        self.steer, self.speed, self.acceleration = steer, speed, acceleration


class Color:
    def __init__(self, r=0, g=0, b=0):
        pass


class _Attr:
    def __init__(self, bp_id):
        self.id = bp_id
        self.attrs = {}

    def has_attribute(self, k):
        return True

    def set_attribute(self, k, v):
        self.attrs[k] = v


class _Lib:
    def filter(self, pattern):
        return [_Attr(pattern)] if pattern.startswith("vehicle.") else []

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

    def stop(self):
        self.cb = None

    def get_transform(self):
        return self.tf


class Vehicle(Actor):
    L = 2.8

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.v = 0.0
        self.steer_rad = 0.0     # CARLA convention: positive = right
        self.ctrl = VehicleControl()

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
        self.target_speed = self.v + (c.throttle * 3.0 - c.brake * 6.0)
        self.steer_rad = c.steer * math.radians(70.0)

    def physics(self, dt):
        dv = np.clip(getattr(self, "target_speed", 0.0) - self.v, -4 * dt, 2 * dt)
        self.v = max(0.0, self.v + dv)
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

    def get_waypoint(self, loc):
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
        a = Vehicle(self, bp, tf) if bp.id.startswith("vehicle.") else Actor(self, bp, tf, attach_to)
        self.actors.append(a)
        return a

    def _sensor_world_tf(self, s):
        p = s.parent.tf
        yaw = math.radians(p.rotation.yaw)
        x = p.location.x + s.tf.location.x * math.cos(yaw) - s.tf.location.y * math.sin(yaw)
        y = p.location.y + s.tf.location.x * math.sin(yaw) + s.tf.location.y * math.cos(yaw)
        return Location(x, y, s.tf.location.z), yaw

    def tick(self):
        dt = self.settings.fixed_delta_seconds or 0.05
        for a in self.actors:
            if isinstance(a, Vehicle):
                a.physics(dt)
        self.frame += 1
        for s in self.actors:
            if s.cb is None:
                continue
            loc, yaw = self._sensor_world_tf(s)
            if s.bp.id == "sensor.other.gnss":
                g = _geo(loc)
                s.cb(_Data(self.frame, latitude=g.latitude, longitude=g.longitude, altitude=g.altitude))
            else:
                rel = self.walls - [loc.x, loc.y]
                rel = rel[np.linalg.norm(rel, axis=1) < 40]
                c, sn = math.cos(yaw), math.sin(yaw)
                sx = rel[:, 0] * c + rel[:, 1] * sn       # world -> sensor (left-handed)
                sy = -rel[:, 0] * sn + rel[:, 1] * c
                z = np.full_like(sx, 1.0 - loc.z)          # 1 m above ground
                raw = np.column_stack([sx, sy, z, np.ones_like(sx)]).astype(np.float32)
                s.cb(_Data(self.frame, raw_data=raw.tobytes()))
        return self.frame


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
