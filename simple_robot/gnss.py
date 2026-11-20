"""Conversions from GNSS (latitude, longitude, altitude) to a local metric frame.

Two projections are provided:

* :class:`CarlaGeoProjection` - the exact inverse of the projection CARLA uses
  in ``carla.Map.transform_to_geolocation`` / ``sensor.other.gnss`` (a scaled
  Mercator projection about the map's geo-reference).  Using it gives a local
  frame that lines up with CARLA world coordinates to sub-millimetre accuracy.
* :class:`LocalTangentPlane` - a standard WGS-84 ECEF -> ENU conversion, which is
  what you would use on a real robot.

Both return (east, north, up) in metres.  East/north form the right-handed
x/y plane used everywhere in the stack.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_EQUA = 6378137.0  # same constant as LibCarla


class CarlaGeoProjection:
    def __init__(self, lat0: float, lon0: float, alt0: float = 0.0):
        self.lat0, self.lon0, self.alt0 = lat0, lon0, alt0
        self.scale = np.cos(np.deg2rad(lat0))
        self._mx0, self._my0 = self._mercator(lat0, lon0)

    def _mercator(self, lat, lon):
        mx = self.scale * np.deg2rad(lon) * EARTH_RADIUS_EQUA
        my = self.scale * EARTH_RADIUS_EQUA * np.log(np.tan((90.0 + np.asarray(lat)) * np.pi / 360.0))
        return mx, my

    def to_enu(self, lat, lon, alt=0.0) -> np.ndarray:
        mx, my = self._mercator(lat, lon)
        return np.array([mx - self._mx0, my - self._my0, alt - self.alt0])

    def from_enu(self, east, north, up=0.0) -> tuple[float, float, float]:
        mx, my = self._mx0 + east, self._my0 + north
        lon = mx * 180.0 / (np.pi * EARTH_RADIUS_EQUA * self.scale)
        lat = 360.0 * np.arctan(np.exp(my / (EARTH_RADIUS_EQUA * self.scale))) / np.pi - 90.0
        return float(lat), float(lon), float(self.alt0 + up)


class LocalTangentPlane:
    A = 6378137.0
    F = 1.0 / 298.257223563
    E2 = F * (2.0 - F)

    def __init__(self, lat0: float, lon0: float, alt0: float = 0.0):
        self.lat0, self.lon0, self.alt0 = lat0, lon0, alt0
        self._ecef0 = self._to_ecef(lat0, lon0, alt0)
        la, lo = np.deg2rad(lat0), np.deg2rad(lon0)
        self._R = np.array([
            [-np.sin(lo), np.cos(lo), 0.0],
            [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
            [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)],
        ])

    def _to_ecef(self, lat, lon, alt):
        la, lo = np.deg2rad(lat), np.deg2rad(lon)
        n = self.A / np.sqrt(1.0 - self.E2 * np.sin(la) ** 2)
        return np.array([
            (n + alt) * np.cos(la) * np.cos(lo),
            (n + alt) * np.cos(la) * np.sin(lo),
            (n * (1.0 - self.E2) + alt) * np.sin(la),
        ])

    def to_enu(self, lat, lon, alt=0.0) -> np.ndarray:
        return self._R @ (self._to_ecef(lat, lon, alt) - self._ecef0)
