# test_station_geometries.py
import itertools
import logging
import os
from pathlib import Path

import numpy as np
import pytest
from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Pick, WaveformStreamID
from obspy.core.event.base import QuantityError
from obspy.core.inventory import Inventory, Network, Site, Station
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel

# --- NEW: plotting imports (non-interactive backend) ---
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
# -------------------------------------------------------

from hypy.event_locator import EventLocator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------- Fixtures & helpers --------------------


@pytest.fixture(scope="session", autouse=True)
def _seed():
    np.random.seed(42)


@pytest.fixture(scope="session")
def taup_model():
    return TauPyModel(model="iasp91")


@pytest.fixture(scope="session")
def true_event_params():
    # (lat, lon, depth[km]), origin time
    return (10.0, 10.0, 15.0), UTCDateTime("2023-01-01T00:00:00.000Z")


def _iasp91_tvel_path():
    import obspy

    obspy_path = Path(obspy.__file__).parents[0].as_posix()
    return os.path.join(obspy_path, "taup/data/iasp91.tvel")


def _km_to_deg_offsets(lat0_deg, d_north_km, d_east_km):
    """Return (dlat_deg, dlon_deg) for small offsets at latitude lat0."""
    dlat = d_north_km / 111.0
    dlon = d_east_km / (111.0 * np.cos(np.radians(lat0_deg)))
    return dlat, dlon


def _circle_geometry(center, radius_km, n, prefix="C"):
    lat0, lon0, _ = center
    sta = {}
    for i, theta in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
        north = radius_km * np.cos(theta)
        east = radius_km * np.sin(theta)
        dlat, dlon = _km_to_deg_offsets(lat0, north, east)
        sta[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return sta


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
    sta = {}
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i, th in enumerate(thetas):
        dn = radius_km * np.cos(th)  # northing from circle center
        de = radius_km * np.sin(th)  # easting from circle center
        dlat, dlon = _km_to_deg_offsets(lat_c, dn, de)
        sta[f"{prefix}{i + 1:02d}"] = (lat_c + dlat, lon_c + dlon, 0.0)
    return sta


def _square_geometry(center, halfside_km=4.0, prefix="Q"):
    lat0, lon0, _ = center
    sta = {}
    offs = [
        (-halfside_km, -halfside_km),
        (-halfside_km, halfside_km),
        (halfside_km, -halfside_km),
        (halfside_km, halfside_km),
    ]
    for i, (dn, de) in enumerate(offs):
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        sta[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return sta


def _line_geometry_NS(center, length_km=60.0, n=6, prefix="L"):
    lat0, lon0, _ = center
    sta = {}
    # equally spaced along N-S line centered on event
    offs = np.linspace(-length_km / 2, length_km / 2, n)
    for i, dn in enumerate(offs):
        dlat, dlon = _km_to_deg_offsets(lat0, dn, 0.0)
        sta[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return sta


def _L_shape_geometry(center, arm_km=40.0, step_km=10.0, prefix="LS"):
    lat0, lon0, _ = center
    sta = {}
    # North arm (including corner)
    for i, dn in enumerate(np.arange(0.0, arm_km + 1e-6, step_km)):
        dlat, dlon = _km_to_deg_offsets(lat0, dn, 0.0)
        sta[f"{prefix}N{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    # East arm (excluding corner to avoid duplicate)
    for j, de in enumerate(np.arange(step_km, arm_km + 1e-6, step_km), start=1):
        dlat, dlon = _km_to_deg_offsets(lat0, 0.0, de)
        sta[f"{prefix}E{j:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return sta


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
    sta = {}

    # How many stations per arm (including the corner once overall).
    # Horizontal arm: positions along +EAST from the corner.
    # Vertical arm:   positions along +NORTH from the corner.
    n_east = int(arm_km // step_km) + 1
    n_north = int(arm_km // step_km) + 1

    # Corner (gap east, gap north) — add once
    dlat_corner, dlon_corner = _km_to_deg_offsets(lat0, gap_km, gap_km)
    corner_lat = lat0 + dlat_corner
    corner_lon = lon0 + dlon_corner
    sta[f"{prefix}_C"] = (corner_lat, corner_lon, 0.0)

    # Horizontal arm: keep latitude at corner (gap north), move east from corner
    for j in range(1, n_east):
        de = gap_km + j * step_km  # east distance from source to point
        # latitude stays at +gap_km, east increases beyond corner
        dlat, dlon = _km_to_deg_offsets(lat0, gap_km, de)
        sta[f"{prefix}E{j:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)

    # Vertical arm: keep longitude at corner (gap east), move north from corner
    for i in range(1, n_north):
        dn = gap_km + i * step_km  # north distance from source to point
        dlat, dlon = _km_to_deg_offsets(lat0, dn, gap_km)
        sta[f"{prefix}N{i:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)

    return sta


def _one_sided_semicircle(
    center, radius_km=25.0, n=8, bearing_center_deg=90.0, prefix="S"
):
    """Stations only to the east (bearing 90°) spanning 180° aperture."""
    lat0, lon0, _ = center
    sta = {}
    start = np.radians(bearing_center_deg - 90.0)
    thetas = np.linspace(start, start + np.pi, n)
    for i, th in enumerate(thetas):
        dn = radius_km * np.cos(th)
        de = radius_km * np.sin(th)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        sta[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return sta


def _cluster_near(center, radius_km=5.0, n=5, prefix="CL", seed=123):
    # random small cluster around event
    lat0, lon0, _ = center
    rng = np.random.default_rng(seed)
    sta = {}
    for i in range(n):
        r = radius_km * np.sqrt(rng.random())
        th = 2 * np.pi * rng.random()
        dn = r * np.cos(th)
        de = r * np.sin(th)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        sta[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return sta


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
    sta = {}

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
        sta[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)

    return sta


def build_inventory(station_dict):
    net = Network(code="NET", start_date=UTCDateTime("2020-01-01"))
    for code, (lat, lon, elev) in station_dict.items():
        net.stations.append(
            Station(
                code=code,
                latitude=lat,
                longitude=lon,
                elevation=elev,
                start_date=UTCDateTime("2020-01-01"),
                site=Site(name=code),
            )
        )
    return Inventory(networks=[net], source="synthetic-geom")


def make_single_event(model, stations, phases, true_loc, true_t, pick_uncertainty=0.01):
    """Create a single Event with synthetic P/S picks for provided stations."""
    lat, lon, depth = true_loc
    ev = Event()
    phase_map = []
    if "P" in phases:
        phase_map.extend(["p", "P"])
    if "S" in phases:
        phase_map.extend(["s", "S"])

    for sta, (sta_lat, sta_lon, elev) in stations.items():
        arrivals = model.get_travel_times_geo(
            source_latitude_in_deg=lat,
            source_longitude_in_deg=lon,
            source_depth_in_km=depth + elev / 1000.0,
            receiver_latitude_in_deg=sta_lat,
            receiver_longitude_in_deg=sta_lon,
            phase_list=phase_map,
        )
        for arr in [a for a in arrivals if a.name in ["p", "s", "P", "S"]]:
            pick_time = true_t + arr.time
            ev.picks.append(
                Pick(
                    time=pick_time,
                    time_errors=QuantityError(uncertainty=0.01, confidence_level=100.0),
                    phase_hint=arr.name,
                    waveform_id=WaveformStreamID(
                        network_code="NET", station_code=sta, channel_code="HHZ"
                    ),
                )
            )
    # store truth for assertion
    ev.true_location = np.array([lat, lon, depth])
    ev.true_origin_time = true_t
    cat = Catalog()
    cat.append(ev)
    return cat


def assert_single_event(loc, t0, event, tol_arc_km, tol_depth_km, tol_time_s):
    true_loc = event.true_location
    true_t = event.true_origin_time.timestamp
    arc_m, _, _ = gps2dist_azimuth(true_loc[0], true_loc[1], loc[0], loc[1])
    arc_km = arc_m / 1000.0
    depth_err = abs(true_loc[2] - loc[2])
    time_err = abs(true_t - t0)
    logger.info(
        f"Geo error: {arc_km:.3f} km, depth err: {depth_err:.3f} km, time err: {time_err:.3f} s"
    )
    assert arc_km <= tol_arc_km
    assert depth_err <= tol_depth_km
    assert time_err <= tol_time_s


# --- NEW: plotting helper -----------------------------------------------------
def _save_geom_plot(
    stations,
    true_loc,
    est_loc,
    geometry_name,
    phases,
    mode,
    engine,
    geo_err_km,
    depth_err_km,
    time_err_s,
):
    """
    Save a 2D lat/lon plot showing station geometry, true epicenter, and solution.
    """
    outdir = Path("test_artifacts") / "station_geometries"
    outdir.mkdir(parents=True, exist_ok=True)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # Stations
    st_lats = [v[0] for v in stations.values()]
    st_lons = [v[1] for v in stations.values()]
    ax.scatter(st_lons, st_lats, marker="^", s=30, label="Stations", zorder=2)

    # True and estimated
    ax.scatter([true_loc[1]], [true_loc[0]], marker="*", s=80, label="True", zorder=4)
    ax.scatter(
        [est_loc[1]],
        [est_loc[0]],
        facecolors="none",
        edgecolors="k",
        marker="o",
        s=70,
        label="Estimated",
        zorder=5,
    )

    # Connect true to estimated
    ax.plot(
        [true_loc[1], est_loc[1]],
        [true_loc[0], est_loc[0]],
        linestyle="--",
        linewidth=1.0,
        zorder=3,
    )

    # Cosmetics
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, linestyle=":")
    ax.set_aspect("equal", adjustable="box")
    phase_tag = "".join(phases)
    ax.set_title(
        f"{geometry_name} | phases={phase_tag} | {mode}/{engine}\n"
        f"geo={geo_err_km:.2f} km, depth={depth_err_km:.2f} km, time={time_err_s:.2f} s"
    )
    ax.legend(loc="best", frameon=True)

    # Save
    fname = f"{geometry_name}_phases-{phase_tag}_{mode}_{engine}.png"
    fig.tight_layout()
    fig.savefig(outdir / fname)
    plt.close(fig)
    logger.info(f"Saved plot to {outdir / fname}")


# -----------------------------------------------------------------------------


# -------------------- Parametrization --------------------

GEOMETRIES = {
    # name: (builder_fn, kwargs, tolerances (arc_km, depth_km, time_s))
    "ring_20km_8sta": (
        _circle_geometry,
        dict(radius_km=20.0, n=8, prefix="R"),
        (2.5, 1.5, 0.6),
    ),
    "ring_offset_20km_8sta": (
        _circle_offset_center,
        dict(radius_km=20.0, n=8, offset_north_km=2.0, offset_east_km=2.0, prefix="RO"),
        (3.0, 2.0, 0.7),
    ),
    "square_10km_4sta": (
        _square_geometry,
        dict(halfside_km=10.0, prefix="Q"),
        (3.0, 2.0, 0.7),
    ),
    "line_NS_6sta": (
        _line_geometry_NS,
        dict(length_km=60.0, n=6, prefix="L"),
        (6.0, 5.0, 0.9),
    ),
    "L_shape_6sta": (
        _L_shape_geometry,
        dict(arm_km=40.0, step_km=20.0, prefix="LS"),
        (4.5, 3.0, 0.8),
    ),
    "L_shape_offset_gap10_outside": (
        _L_shape_offset_around,
        dict(arm_km=40.0, step_km=10.0, gap_km=10.0, prefix="LSO"),
        # tolerances roughly similar to the standard L-shape; adjust as you like
        (4.5, 3.0, 0.8),
    ),
    "L_shape_offset_gap10_inside": (
        _L_shape_offset_around,
        dict(arm_km=40.0, step_km=10.0, gap_km=-10.0, prefix="LSO"),
        # tolerances roughly similar to the standard L-shape; adjust as you like
        (4.5, 3.0, 0.8),
    ),
    "one_sided_east": (
        _one_sided_semicircle,
        dict(radius_km=25.0, n=8, bearing_center_deg=90.0, prefix="S"),
        (6.0, 4.5, 0.9),
    ),
    "cluster_near_5": (
        _cluster_near,
        dict(radius_km=5.0, n=5, prefix="C5"),
        (7.0, 6.0, 1.0),
    ),
    "cluster_near_10": (
        _cluster_near,
        dict(radius_km=10.0, n=10, prefix="C10"),
        (7.0, 6.0, 1.0),
    ),
    "cluster_one_side_east": (
        _cluster_one_side,
        dict(
            n=10,
            r_min_km=8.0,
            r_max_km=45.0,
            bearing_center_deg=90.0,  # cluster to the EAST of the event
            aperture_deg=120.0,  # 120° wide sector
            seed=42,
            prefix="CSE",
        ),
        (6.5, 5.0, 1.0),  # tolerances: tune as you see fit
    ),
    "cluster_one_side_east_far": (
        _cluster_one_side,
        dict(
            n=10,
            r_min_km=100.0,
            r_max_km=200.0,
            bearing_center_deg=90.0,  # cluster to the EAST of the event
            aperture_deg=120.0,  # 120° wide sector
            seed=42,
            prefix="CSEF",
        ),
        (6.5, 5.0, 1.0),  # tolerances: tune as you see fit
    ),
}


# build parameter grid and nice IDs
_PARAMS = list(
    itertools.product(
        list(GEOMETRIES.keys()),
        [["P"], ["S"], ["P", "S"]],
        ["ray", "table"],
        ["taup", "pyrocko"],
    )
)


def _phase_id(phases):
    return "+".join(phases)


_IDS = [
    f"{geom}|phases={_phase_id(phs)}|{mode}/{eng}" for (geom, phs, mode, eng) in _PARAMS
]


@pytest.mark.parametrize("geometry_name, phases, mode, engine", _PARAMS, ids=_IDS)
def test_event_location_vs_geometry(
    taup_model, true_event_params, geometry_name, phases, mode, engine
):
    (true_loc, true_t) = true_event_params
    builder, kwargs, (tol_arc_km, tol_depth_km, tol_time_s) = GEOMETRIES[geometry_name]

    # build stations for this geometry
    stations = builder(true_loc, **kwargs)
    inventory = build_inventory(stations)

    # synthesize single event with picks
    catalog = make_single_event(taup_model, stations, phases, true_loc, true_t)

    # configure locator
    model_path = _iasp91_tvel_path()
    if mode == "ray":
        locator = EventLocator(model_path, phases=phases, mode="ray", engine=engine)
    else:
        locator = EventLocator(
            model_path,
            phases=phases,
            mode="table",
            engine=engine,
            max_distance=250.0,
            distance_step=1.0,
            max_depth=20.0,
            depth_step=1.0,
        )

    # run
    locator.load_data(inventory, catalog)
    out = locator.locate(max_iterations=1000, trial_depth=5.0, solver="lsqr")

    # extract final origin
    e = out[0]
    est_lat = e.origins[-1].latitude
    est_lon = e.origins[-1].longitude
    est_depth_km = e.origins[-1].depth / 1000.0
    est_time = e.origins[-1].time.timestamp

    arc_m, _, _ = gps2dist_azimuth(true_loc[0], true_loc[1], est_lat, est_lon)
    geo_err_km = arc_m / 1000.0
    depth_err_km = abs(true_loc[2] - est_depth_km)
    time_err_s = abs(catalog[0].true_origin_time.timestamp - est_time)

    _save_geom_plot(
        stations=stations,
        true_loc=true_loc,
        est_loc=(est_lat, est_lon),
        geometry_name=geometry_name,
        phases=phases,
        mode=mode,
        engine=engine,
        geo_err_km=geo_err_km,
        depth_err_km=depth_err_km,
        time_err_s=time_err_s,
    )
    # ----------------------------------------------------------

    # assertions (tolerances reflect geometry strength)
    assert_single_event(
        (est_lat, est_lon, est_depth_km),
        est_time,
        catalog[0],
        tol_arc_km,
        tol_depth_km,
        tol_time_s,
    )
