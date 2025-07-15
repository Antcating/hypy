from numba import njit
import numpy as np

@njit(cache=True)
def gps2dist_azimuth_numba(lat1, lon1, lat2, lon2, R=6371.0e3):
    """
    Compute distance (meters), azimuth (degrees), and back azimuth (degrees)
    between two lat/lon points using spherical Earth approximation.

    Parameters:
    - lat1, lon1: event (deg)
    - lat2, lon2: station (deg)

    Returns:
    - distance (meters), azimuth (deg), back_azimuth (deg)
    """
    # Convert to radians
    φ1 = np.radians(lat1)
    φ2 = np.radians(lat2)
    Δφ = φ2 - φ1
    Δλ = np.radians(lon2 - lon1)

    # Haversine distance
    a = np.sin(Δφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    # Azimuth
    y = np.sin(Δλ) * np.cos(φ2)
    x = np.cos(φ1) * np.sin(φ2) - np.sin(φ1) * np.cos(φ2) * np.cos(Δλ)
    azimuth = np.degrees(np.arctan2(y, x)) % 360

    # Back azimuth
    y_back = -np.sin(Δλ) * np.cos(φ1)
    x_back = np.cos(φ2) * np.sin(φ1) - np.sin(φ2) * np.cos(φ1) * np.cos(-Δλ)
    back_azimuth = np.degrees(np.arctan2(y_back, x_back)) % 360

    return distance, azimuth, back_azimuth

@njit(cache=True)
def compute_spatial_derivatives(
    takeoff_rad: float, az_deg: float, vel: float
) -> tuple:
    """
    Calculate spatial derivatives for travel-time, ray-parameter, and takeoff angle.
    """
    dt_dr = np.sin(takeoff_rad) / vel
    dt_dz = -np.cos(takeoff_rad) / vel
    az_rad = np.deg2rad(az_deg)
    dt_dx = -dt_dr * np.cos(az_rad)
    dt_dy = -dt_dr * np.sin(az_rad)

    return dt_dr, dt_dz, dt_dx, dt_dy