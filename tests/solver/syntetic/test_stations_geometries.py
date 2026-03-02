# test_station_geometries.py
import logging
import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from obspy import UTCDateTime  # type: ignore[import-untyped]
from obspy.core.event import (  # type: ignore[import-untyped]
    Catalog,
    Event,
    Pick,
    WaveformStreamID,
)
from obspy.core.event.base import QuantityError  # type: ignore[import-untyped]
from obspy.core.inventory import (  # type: ignore[import-untyped]
    Inventory,
    Network,
    Site,
    Station,
)
from obspy.geodetics import gps2dist_azimuth  # type: ignore[import-untyped]
from obspy.taup import TauPyModel  # type: ignore[import-untyped]

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------
from hypy.locator import EventLocator
from tests.solver.syntetic.utils.geometry_generator import (
    _circle_geometry,
    _circle_offset_center,
    _cluster_near,
    _cluster_one_side,
    _elevation_variation_geometry,
    _L_shape_geometry,
    _L_shape_offset_around,
    _line_geometry_NS,
    _one_sided_semicircle,
    _pair_one_side_random,
    _pair_opposite,
    _pair_orthogonal_equal_radius,
    _pair_same_az_near_far,
    _sparse_geometry,
    _square_geometry,
    _uniform_grid_geometry,
)

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
    return (9.85, 9.85, 10.0), UTCDateTime("2023-01-01T00:00:00.000Z")


def _iasp91_tvel_path():
    import obspy

    obspy_path = Path(obspy.__file__).parents[0].as_posix()
    return os.path.join(obspy_path, "taup/data/iasp91.tvel")


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

    ax.set_xlim(
        min([est_loc[1], true_loc[1], min(st_lons)]) - 0.1,
        max([est_loc[1], true_loc[1], max(st_lons)]) + 0.1,
    )
    ax.set_ylim(
        min([est_loc[0], true_loc[0], min(st_lats)]) - 0.1,
        max([est_loc[0], true_loc[0], max(st_lats)]) + 0.1,
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
    "uniform_grid_3x3": (
        _uniform_grid_geometry,
        dict(grid_size=3, spacing_km=10.0, prefix="UG"),
        (2.5, 1.5, 0.6),
    ),
    "sparse_3sta": (
        _sparse_geometry,
        dict(n=3, spread_km=50.0, prefix="SP"),
        (10.0, 8.0, 1.5),
    ),
    "elevation_variation": (
        _elevation_variation_geometry,
        dict(n=6, radius_km=20.0, prefix="EL"),
        (3.0, 2.0, 0.7),
    ),
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
    "pair_opposite_20km_az0": (
        _pair_opposite,
        dict(radius_km=20.0, azimuth_deg=10.0, prefix="PO"),
        (8.0, 8.0, 1.2),
    ),
    "pair_same_az_near_far_E": (
        _pair_same_az_near_far,
        dict(r_near_km=10.0, r_far_km=30.0, azimuth_deg=90.0, prefix="PA"),
        (9.0, 10.0, 1.3),
    ),
    "pair_orthogonal_equal_20km": (
        _pair_orthogonal_equal_radius,
        dict(radius_km=20.0, azimuth0_deg=0.0, prefix="POE"),
        (7.0, 7.0, 1.1),
    ),
    "pair_one_side_random_E_sector": (
        _pair_one_side_random,
        dict(
            r_min_km=8.0,
            r_max_km=35.0,
            bearing_center_deg=90.0,
            aperture_deg=60.0,
            seed=7,
            prefix="PRS",
        ),
        (10.0, 12.0, 1.5),
    ),
}


# build parameter grid and nice IDs
ALL_PHASE_SETS = [["P"], ["S"], ["P", "S"]]

# Which geometries are exactly 2-station?
TWO_STATION_GEOMS = {
    "pair_opposite_20km_az0",
    "pair_same_az_near_far_E",
    "pair_orthogonal_equal_20km",
    "pair_one_side_random_E_sector",
}
THREE_STATION_GEOMS = {
    "sparse_3sta",
}


def _phase_id(phs):
    return "+".join(phs)


_PARAMS = []
for geom_name in GEOMETRIES.keys():
    allowed_phase_sets = (
        [["P", "S"]]
        if (geom_name in TWO_STATION_GEOMS or geom_name in THREE_STATION_GEOMS)
        else ALL_PHASE_SETS
    )
    for phases in allowed_phase_sets:
        for mode in ["ray", "table"]:
            for eng in ["taup", "pyrocko"]:
                _PARAMS.append((geom_name, phases, mode, eng))

_IDS = [
    f"{geom}|phases={_phase_id(phs)}|{mode}/{eng}" for (geom, phs, mode, eng) in _PARAMS
]


# -------------------- The Test Itself --------------------
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
            max_depth=50.0,
            depth_step=2.0,
        )

    # run
    locator.load(inventory, catalog)
    out, _ = locator.locate(max_iterations=1000, trial_depth=5.0, solver="lsqr")

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
