import numpy as np


def _km_to_deg_offsets(lat0_deg, d_north_km, d_east_km):
    """Return (dlat_deg, dlon_deg) for small offsets at latitude lat0."""
    dlat = d_north_km / 111.0
    dlon = d_east_km / (111.0 * np.cos(np.radians(lat0_deg)))
    return dlat, dlon


def _circle_geometry(center, radius_km, n, prefix="C"):
    lat0, lon0, _ = center
    locations = {}
    for i, theta in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
        north = radius_km * np.cos(theta)
        east = radius_km * np.sin(theta)
        dlat, dlon = _km_to_deg_offsets(lat0, north, east)
        locations[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return locations


def _circle_offset_center(
    center, radius_km=20.0, n=8, offset_north_km=5.0, offset_east_km=5.0, prefix="RO"
):
    """
    Stations on a circle of radius `radius_km`, but the circle's center is
    offset from the event location by (offset_east_km, offset_north_km),
    so the origin is NOT equidistant to the stations.
    """
    lat_ev, lon_ev, _ = center

    # Compute the circle center (offset from the event)
    dlat_c, dlon_c = _km_to_deg_offsets(lat_ev, offset_north_km, offset_east_km)
    lat_c = lat_ev + dlat_c
    lon_c = lon_ev + dlon_c

    # Place stations evenly around the (offset) circle center
    locations = {}
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i, th in enumerate(thetas):
        dn = radius_km * np.cos(th)  # northing from circle center
        de = radius_km * np.sin(th)  # easting from circle center
        dlat, dlon = _km_to_deg_offsets(lat_c, dn, de)
        locations[f"{prefix}{i + 1:02d}"] = (lat_c + dlat, lon_c + dlon, 0.0)
    return locations


def _square_geometry(center, halfside_km=4.0, prefix="Q"):
    lat0, lon0, _ = center
    locations = {}
    offs = [
        (-halfside_km, -halfside_km),
        (-halfside_km, halfside_km),
        (halfside_km, -halfside_km),
        (halfside_km, halfside_km),
    ]
    for i, (dn, de) in enumerate(offs):
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        locations[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return locations


def _line_geometry_NS(center, length_km=60.0, n=6, prefix="L"):
    lat0, lon0, _ = center
    locations = {}
    # equally spaced along N-S line centered on event
    offs = np.linspace(-length_km / 2, length_km / 2, n)
    for i, dn in enumerate(offs):
        dlat, dlon = _km_to_deg_offsets(lat0, dn, 0.0)
        locations[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return locations


def _L_shape_geometry(center, arm_km=40.0, step_km=10.0, prefix="LS"):
    lat0, lon0, _ = center
    locations = {}
    # North arm (including corner)
    for i, dn in enumerate(np.arange(0.0, arm_km + 1e-6, step_km)):
        dlat, dlon = _km_to_deg_offsets(lat0, dn, 0.0)
        locations[f"{prefix}N{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    # East arm (excluding corner to avoid duplicate)
    for j, de in enumerate(np.arange(step_km, arm_km + 1e-6, step_km), start=1):
        dlat, dlon = _km_to_deg_offsets(lat0, 0.0, de)
        locations[f"{prefix}E{j:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return locations


def _L_shape_offset_around(
    center, arm_km=40.0, step_km=10.0, gap_km=10.0, prefix="LSO"
):
    """
    Build an 'L' shape whose arms do NOT pass through the source.
    - One horizontal arm: at +gap_km to the NORTH of the source, extending EAST.
    - One vertical arm:   at +gap_km to the EAST  of the source, extending NORTH.
    The L-corner is at (gap east, gap north) wrt the source. The source sits
    inside the L (south-west of the corner), not on either arm.
    """
    lat0, lon0, _ = center
    locations = {}

    # How many stations per arm (including the corner once overall).
    # Horizontal arm: positions along +EAST from the corner.
    # Vertical arm:   positions along +NORTH from the corner.
    n_east = int(arm_km // step_km) + 1
    n_north = int(arm_km // step_km) + 1

    # Corner (gap east, gap north) — add once
    dlat_corner, dlon_corner = _km_to_deg_offsets(lat0, gap_km, gap_km)
    corner_lat = lat0 + dlat_corner
    corner_lon = lon0 + dlon_corner
    locations[f"{prefix}_C"] = (corner_lat, corner_lon, 0.0)

    # Horizontal arm: keep latitude at corner (gap north), move east from corner
    for j in range(1, n_east):
        de = gap_km + j * step_km  # east distance from source to point
        # latitude stays at +gap_km, east increases beyond corner
        dlat, dlon = _km_to_deg_offsets(lat0, gap_km, de)
        locations[f"{prefix}E{j:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)

    # Vertical arm: keep longitude at corner (gap east), move north from corner
    for i in range(1, n_north):
        dn = gap_km + i * step_km  # north distance from source to point
        dlat, dlon = _km_to_deg_offsets(lat0, dn, gap_km)
        locations[f"{prefix}N{i:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)

    return locations


def _one_sided_semicircle(
    center, radius_km=25.0, n=8, bearing_center_deg=90.0, prefix="S"
):
    """Stations only to the east (bearing 90°) spanning 180° aperture."""
    lat0, lon0, _ = center
    locations = {}
    start = np.radians(bearing_center_deg - 90.0)
    thetas = np.linspace(start, start + np.pi, n)
    for i, th in enumerate(thetas):
        dn = radius_km * np.cos(th)
        de = radius_km * np.sin(th)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        locations[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return locations


def _cluster_near(center, radius_km=5.0, n=5, prefix="CL", seed=123):
    # random small cluster around event
    lat0, lon0, _ = center
    rng = np.random.default_rng(seed)
    locations = {}
    for i in range(n):
        r = radius_km * np.sqrt(rng.random())
        th = 2 * np.pi * rng.random()
        dn = r * np.cos(th)
        de = r * np.sin(th)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        locations[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return locations


def _cluster_one_side(
    center,
    n=8,
    r_min_km=5.0,
    r_max_km=40.0,
    bearing_center_deg=90.0,  # 0=N, 90=E, 180=S, 270=W
    aperture_deg=120.0,  # total angular width of the sector
    seed=123,
    prefix="CS",
):
    """
    Build a random cluster of 'n' stations confined to a sector on ONE SIDE
    of the event (like land-only stations for an offshore earthquake).

    Parameters
    ----------
    center : (lat, lon, depth)
        Event location; stations are distributed around this point.
    n : int
        Number of stations.
    r_min_km, r_max_km : float
        Radial distance bounds (km) from the event to stations.
    bearing_center_deg : float
        Direction (from the event) pointing to the middle of the sector.
        0=N, 90=E, 180=S, 270=W.
    aperture_deg : float
        Total angular width of the sector in degrees.
    seed : int
        RNG seed for reproducibility.
    prefix : str
        Station code prefix.

    Returns
    -------
    dict: {code: (lat, lon, elev_m)}
    """
    lat0, lon0, _ = center
    rng = np.random.default_rng(seed)
    locations = {}

    half_ap = np.deg2rad(aperture_deg / 2.0)
    th_center = np.deg2rad(bearing_center_deg)

    for i in range(n):
        # Uniform in a ring sector: radius ~ sqrt(U) for area-uniform sampling
        r = np.sqrt(rng.uniform(r_min_km**2, r_max_km**2))
        # Angle within sector around bearing_center
        theta = th_center + rng.uniform(-half_ap, half_ap)

        dn = r * np.cos(theta)  # north component (km)
        de = r * np.sin(theta)  # east  component (km)

        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        locations[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)

    return locations


def _pair_opposite(center, radius_km=20.0, azimuth_deg=0.0, prefix="PO"):
    """
    Two stations at the same range, opposite azimuths (±azimuth_deg).
    Good for epicenter trade-off, weak depth.
    """
    lat0, lon0, _ = center
    locations = {}
    for i, sign in enumerate([+1, -1], start=1):
        az = np.deg2rad(sign * azimuth_deg)
        dn = radius_km * np.cos(az)
        de = radius_km * np.sin(az)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        locations[f"{prefix}{i:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return locations


def _pair_same_az_near_far(
    center, r_near_km=10.0, r_far_km=30.0, azimuth_deg=90.0, prefix="PA"
):
    """
    Two stations on the same azimuth from the source but at different ranges (near/far).
    Strengthens time–depth constraints along one ray family.
    """
    lat0, lon0, _ = center
    locations = {}
    az = np.deg2rad(azimuth_deg)
    for i, r in enumerate([r_near_km, r_far_km], start=1):
        dn = r * np.cos(az)
        de = r * np.sin(az)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        locations[f"{prefix}{i:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return locations


def _pair_orthogonal_equal_radius(
    center, radius_km=20.0, azimuth0_deg=0.0, prefix="POE"
):
    """
    Two stations 90° apart at equal range (azimuth0 and azimuth0+90).
    Better azimuthal coverage than colinear pairs.
    """
    lat0, lon0, _ = center
    locations = {}
    for i, azdeg in enumerate([azimuth0_deg, azimuth0_deg + 90.0], start=1):
        az = np.deg2rad(azdeg % 360.0)
        dn = radius_km * np.cos(az)
        de = radius_km * np.sin(az)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        locations[f"{prefix}{i:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return locations


def _uniform_grid_geometry(center, grid_size=3, spacing_km=10.0, prefix="UG"):
    """
    Stations arranged in a uniform grid around the event.
    grid_size: number of stations per side (e.g., 3 for 3x3)
    spacing_km: distance between adjacent stations
    """
    lat0, lon0, _ = center
    locations = {}
    offset = (grid_size - 1) / 2
    for i in range(grid_size):
        for j in range(grid_size):
            dn = (i - offset) * spacing_km
            de = (j - offset) * spacing_km
            dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
            locations[f"{prefix}{i + 1}{j + 1}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return locations


def _sparse_geometry(center, n=3, spread_km=50.0, prefix="SP"):
    """
    Sparse geometry: few stations spread far apart.
    """
    lat0, lon0, _ = center
    locations = {}
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i, th in enumerate(angles):
        dn = spread_km * np.cos(th)
        de = spread_km * np.sin(th)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        locations[f"{prefix}{i + 1}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return locations


def _elevation_variation_geometry(center, n=6, radius_km=20.0, prefix="EL"):
    """
    Stations on a circle, each with different elevation.
    """
    lat0, lon0, _ = center
    locations = {}
    elevations = np.linspace(0, 1000, n)  # 0 to 1000 meters
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i, (th, elev) in enumerate(zip(angles, elevations)):
        dn = radius_km * np.cos(th)
        de = radius_km * np.sin(th)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        locations[f"{prefix}{i + 1}"] = (lat0 + dlat, lon0 + dlon, elev)
    return locations


def _pair_one_side_random(
    center,
    r_min_km=8.0,
    r_max_km=35.0,
    bearing_center_deg=90.0,
    aperture_deg=60.0,
    seed=7,
    prefix="PRS",
):
    """
    Two stations randomly placed within a sector on one side of the source (e.g., land-only).
    """
    lat0, lon0, _ = center
    rng = np.random.default_rng(seed)
    locations = {}
    half_ap = np.deg2rad(aperture_deg / 2.0)
    th_center = np.deg2rad(bearing_center_deg)
    for i in range(2):
        r = np.sqrt(rng.uniform(r_min_km**2, r_max_km**2))
        theta = th_center + rng.uniform(-half_ap, +half_ap)
        dn = r * np.cos(theta)
        de = r * np.sin(theta)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        locations[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return locations
