from numba import njit
import numpy as np


@njit(cache=True)
def gps2dist_azimuth_nb(lat1, lon1, lat2, lon2, R=6371.0e3):
    """
    Compute distance (meters), azimuth (degrees), and back azimuth (degrees)
    between two lat/lon points using spherical Earth approximation.

    @This function is a Numba-optimized version of Obspy's gps2dist_azimuth.

    Parameters:
    - lat1, lon1: event (deg)
    - lat2, lon2: station (deg)

    Returns:
    - distance (meters), azimuth (deg), back_azimuth (deg)
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

    # Back azimuth
    y_back = -np.sin(d_lon_rad) * np.cos(lan1_rad)
    x_back = np.cos(lat2_rad) * np.sin(lan1_rad) - np.sin(lat2_rad) * np.cos(
        lan1_rad
    ) * np.cos(-d_lon_rad)
    back_azimuth = np.degrees(np.arctan2(y_back, x_back)) % 360

    return distance, azimuth, back_azimuth


@njit(cache=True)
def get_spatial_slowness_1d(takeoff_rad: float, az_deg: float, vel: float):
    """
    Spatial derivatives of travel time T for a 1D isotropic model.

    Axes & conventions:
      - x = East (km), y = North (km), z = depth (km, positive DOWN).
      - takeoff_rad: angle from DOWN (TauP).
      - az_deg: azimuth from source to receiver, clockwise from North.
      - vel: phase velocity (km/s).

    Returns (s/km): dt_dr (horizontal magnitude), dt_dz, dt_dx (East), dt_dy (North)
    """
    az_rad = np.deg2rad(az_deg)

    # Horizontal slowness magnitude
    dt_dr = np.sin(takeoff_rad) / vel

    # Depth derivative (depth positive down)
    dt_dz = np.cos(takeoff_rad) / vel

    # Project horizontal slowness using azimuth-from-North (cw)
    dt_dx = dt_dr * np.sin(az_rad)  # East
    dt_dy = dt_dr * np.cos(az_rad)  # North

    return (dt_dr, dt_dx, dt_dy, dt_dz)


# ----- Derivative wrt source from first ray segment -----
@njit(cache=True)
def dT_dsource_from_trace(path_xyz, v_source, nseg=3):
    """
    path_xyz: (N,3) points from SOURCE -> RECEIVER.
    v_source: km/s at the source (grid value).
    Returns (dT/dx_s, dT/dy_s, dT/dz_s) in s/km.
    """
    if path_xyz.shape[0] < 2:
        raise ValueError("Ray path too short")

    nseg = max(1, min(nseg, path_xyz.shape[0] - 1))

    d = np.zeros(3, dtype=np.float64)
    for i in range(nseg):
        seg = path_xyz[i + 1] - path_xyz[i]
        n = np.linalg.norm(seg)
        if n > 0:
            d += seg / n

    nd = np.linalg.norm(d)
    if nd == 0:
        raise ValueError("Degenerate ray direction near source")

    u_sr = d / nd  # unit direction: source -> receiver
    p = u_sr / v_source  # slowness vector at source (s/km)
    return (-p[0], -p[1], -p[2])  # derivative wrt SOURCE location
