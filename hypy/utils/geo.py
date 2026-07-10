"""Numba-optimized geodesy utilities for common geospatial functions."""

import numpy as np
from numba import njit  # type: ignore[import-untyped]


@njit(cache=True)
def gps2dist_azimuth_nb(
    lat1: float, lon1: float, lat2: float, lon2: float, R: float = 6371.0e3
) -> tuple[float, float, float]:
    """
    Compute distance, azimuth, and back azimuth between two lat/lon points.

    Compute distance (meters), azimuth (degrees), and back azimuth (degrees)
    between two lat/lon points using spherical Earth approximation.

    This function is a Numba-optimized version of obspy's gps2dist_azimuth.

    Parameters
    ----------
    lat1 : float
        Latitude of the first point in degrees.
    lon1 : float
        Longitude of the first point in degrees.
    lat2 : float
        Latitude of the second point in degrees.
    lon2 : float
        Longitude of the second point in degrees.
    R : float, optional
        Radius of the Earth in meters, by default 6371.0e3

    Returns
    -------
    tuple[float, float, float]
        Distance (meters), azimuth (degrees), and back azimuth (degrees)
    """
    # Convert to radians
    lan1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    d_lat_rad = lat2_rad - lan1_rad
    d_lon_rad = np.radians(lon2 - lon1)

    # Haversine distance
    a = (
        np.sin(d_lat_rad / 2) ** 2
        + np.cos(lan1_rad) * np.cos(lat2_rad) * np.sin(d_lon_rad / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    # Azimuth
    y = np.sin(d_lon_rad) * np.cos(lat2_rad)
    x = np.cos(lan1_rad) * np.sin(lat2_rad) - np.sin(lan1_rad) * np.cos(
        lat2_rad
    ) * np.cos(d_lon_rad)
    azimuth = np.degrees(np.arctan2(y, x)) % 360
    # Normalize to [0, 360)
    if azimuth >= 360.0:
        azimuth = 0.0

    # Back azimuth
    y_back = -np.sin(d_lon_rad) * np.cos(lan1_rad)
    x_back = np.cos(lat2_rad) * np.sin(lan1_rad) - np.sin(lat2_rad) * np.cos(
        lan1_rad
    ) * np.cos(-d_lon_rad)
    back_azimuth = np.degrees(np.arctan2(y_back, x_back)) % 360
    # Normalize to [0, 360)
    if back_azimuth >= 360.0:
        back_azimuth = 0.0

    return distance, azimuth, back_azimuth


@njit(cache=True)
def get_analytical_derivative_1d(takeoff_rad: float, az_deg: float, vel: float):
    """
    Get analytical spatial derivatives for a specific ray.

    Given a takeoff angle, azimuth, and phase velocity, compute the spatial slowness
    derivatives (dt/dx, dt/dy, dt/dz) for a ray path in a 1D velocity model,
    assuming a flat-Earth approximation.

    Parameters
    ----------
    takeoff_rad : float
        Takeoff angle in radians (angle between ray path and vertical, positive down).
    az_deg : float
        Azimuth from source to receiver in degrees (clockwise from North).
    vel : float
        Velocity of the given layer in km/s.

    Returns
    -------
    dt_dr : float
        Derivative of travel time with respect to path distance (s/km).
    dt_dx : float
        Derivative of travel time with respect to source x-coordinate (s/km).
    dt_dy : float
        Derivative of travel time with respect to source y-coordinate (s/km).
    dt_dz : float
        Derivative of travel time with respect to source depth (s/km, positive in).
    """
    az_rad = np.deg2rad(az_deg)

    # Horizontal slowness magnitude
    dt_dr = -np.sin(takeoff_rad) / vel

    # Depth derivative (depth positive down)
    dt_dz = -np.cos(takeoff_rad) / vel

    # Project horizontal slowness using azimuth-from-North (cw)
    dt_dx = dt_dr * np.sin(az_rad)  # East
    dt_dy = dt_dr * np.cos(az_rad)  # North

    return (dt_dr, dt_dx, dt_dy, dt_dz)
