from __future__ import annotations

import os
import random
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import zlib
import numpy as np
from obspy import UTCDateTime  # type: ignore[import-untyped]
from obspy.core.event import Catalog, Event, Pick, WaveformStreamID  # type: ignore[import-untyped]
from obspy.core.event.base import QuantityError  # type: ignore[import-untyped]
from obspy.core.inventory import Inventory, Network, Site, Station  # type: ignore[import-untyped]
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees  # type: ignore[import-untyped]
import matplotlib

from hypy.logger import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tests.solver.syntetic.utils.test_cache_utils import load_cache, save_cache
from tests.solver.syntetic.utils.test_output_utils import _save_results_json


# -------------------- Dataclasses --------------------


@dataclass(frozen=True)
class Scenario:
    geometry_name: str
    phases: list[str]
    mode: str
    engine: str
    solve_mode: str  # "batch" | "single"
    solver: str  # "lsqr" | "svd"
    bias: float
    n_stations: int
    n_events: int
    seed: int


@dataclass
class DataBundle:
    inventory: Inventory
    stations_used: dict[str, tuple[float, float, float]]
    true_events: list[tuple[tuple[float, float, float], UTCDateTime]]
    cached_catalog: Catalog
    station_time_offsets_s: dict[str, float]


# -------------------- Determinism helpers --------------------


@contextmanager
def deterministic_seed(seed: int):
    """
    Freeze global RNGs used by many libs (python.random & legacy np.random)
    for the duration of the block, then restore their prior state.
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


# -------------------- Geometry helpers --------------------


def _km_to_deg_offsets(lat0_deg, d_north_km, d_east_km):
    dlat = d_north_km / 111.0
    dlon = d_east_km / (111.0 * max(np.cos(np.radians(lat0_deg)), 1e-9))
    return dlat, dlon


def _circle_geometry(center, radius_km, n, prefix="R"):
    lat0, lon0, _ = center
    D = {}
    for i, th in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
        dn = radius_km * np.cos(th)
        de = radius_km * np.sin(th)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        D[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return D


def _uniform_grid_geometry(center, grid_size=3, spacing_km=10.0, prefix="UG"):
    lat0, lon0, _ = center
    D = {}
    offset = (grid_size - 1) / 2
    for i in range(grid_size):
        for j in range(grid_size):
            dn = (i - offset) * spacing_km
            de = (j - offset) * spacing_km
            dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
            D[f"{prefix}{i + 1}{j + 1}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return D


def _L_shape_geometry(center, arm_km=40.0, step_km=10.0, prefix="LS"):
    lat0, lon0, _ = center
    D = {}
    for i, dn in enumerate(np.arange(0.0, arm_km + 1e-6, step_km)):
        dlat, dlon = _km_to_deg_offsets(lat0, dn, 0.0)
        D[f"{prefix}N{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    for j, de in enumerate(np.arange(step_km, arm_km + 1e-6, step_km), start=1):
        dlat, dlon = _km_to_deg_offsets(lat0, 0.0, de)
        D[f"{prefix}E{j:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return D


def _sparse_geometry(center, n=5, spread_km=60.0, prefix="SP"):
    lat0, lon0, _ = center
    D = {}
    for i, th in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
        dn = spread_km * np.cos(th)
        de = spread_km * np.sin(th)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        D[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return D


def _cluster_near(center, radius_km=20.0, n=6, prefix="CL", seed=123):
    lat0, lon0, _ = center
    rng = np.random.default_rng(seed)
    D = {}
    for i in range(n):
        r = radius_km * np.sqrt(rng.random())
        th = 2 * np.pi * rng.random()
        dn = r * np.cos(th)
        de = r * np.sin(th)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        D[f"{prefix}{i + 1:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return D


GEOMS = {
    "uniform_grid": (
        _uniform_grid_geometry,
        dict(grid_size=3, spacing_km=10.0, prefix="UG"),
        (3.5, 2.5),
    ),
    "ring": (_circle_geometry, dict(radius_km=20.0, n=8, prefix="R"), (3.5, 2.5)),
    "L_shape": (
        _L_shape_geometry,
        dict(arm_km=40.0, step_km=10.0, prefix="LS"),
        (5.0, 3.5),
    ),
    "sparse": (_sparse_geometry, dict(n=5, spread_km=60.0, prefix="SP"), (9.0, 8.0)),
    "cluster": (_cluster_near, dict(n=6, radius_km=20.0, prefix="CL"), (9.0, 8.0)),
}


def adjust_geometry_kwargs(builder, kwargs, n_stations):
    o = dict(kwargs)
    if "n" in o:
        o["n"] = int(n_stations)
        return o
    if builder is _uniform_grid_geometry:
        gs = int(np.ceil(np.sqrt(n_stations)))
        o["grid_size"] = max(2, gs)
        return o
    if builder in (_circle_geometry, _cluster_near, _sparse_geometry):
        o["n"] = int(n_stations)
        return o
    if builder is _L_shape_geometry:
        arm = float(o.get("arm_km", 40.0))
        target = max(3, int(n_stations))
        step = max(1.0, (2.0 * arm) / max(1, target - 1))
        o["step_km"] = step
        return o
    return o


# -------------------- Synthetic data builders --------------------


def _iasp91_tvel_path():
    import obspy

    obspy_path = Path(obspy.__file__).parents[0].as_posix()
    return os.path.join(obspy_path, "taup/data/iasp91.tvel")


def build_inventory_with_variations(
    station_dict, start_date, loc_jitter_km=0.0, rng=None
):
    if rng is None:
        rng = np.random.default_rng()
    net = Network(code="NET", start_date=start_date)
    km_per_deg_lat = 111.32
    for code, (lat, lon, elev) in station_dict.items():
        dlat = dlon = 0.0
        if loc_jitter_km > 0.0:
            theta = rng.uniform(0.0, 2 * np.pi)
            dy = loc_jitter_km * np.sin(theta)
            dx = loc_jitter_km * np.cos(theta)
            dlat = dy / km_per_deg_lat
            km_per_deg_lon = km_per_deg_lat * max(np.cos(np.deg2rad(lat)), 1e-6)
            dlon = dx / km_per_deg_lon
        net.stations.append(
            Station(
                code=code,
                latitude=lat + dlat,
                longitude=lon + dlon,
                elevation=elev,
                start_date=start_date,
                site=Site(name=code),
            )
        )
    return Inventory(networks=[net], source="synthetic-geom-var")


def make_event_list(n_events=50, spacing_s=120.0, spread_km=8.0):
    base_t = UTCDateTime("2023-01-01T00:00:00.000Z")
    lat0, lon0, depth0 = 10.0, 10.05, 4.0
    phi = np.pi * (3.0 - np.sqrt(5.0))
    evs = []
    for i in range(n_events):
        r = spread_km * np.sqrt(i / max(1, n_events - 1))
        th = i * phi
        dn = r * np.cos(th)
        de = r * np.sin(th)
        dlat, dlon = _km_to_deg_offsets(lat0, dn, de)
        depth = depth0 + (i % 2)
        t = base_t + i * spacing_s
        evs.append(((lat0 + dlat, lon0 + dlon, depth), t))
    return evs, base_t


def make_catalog_with_biases(
    taup_model,
    stations,
    true_events,
    phases=("P", "S"),
    pick_uncertainty=0.01,
    station_time_offsets_s=None,
):
    if station_time_offsets_s is None:
        station_time_offsets_s = {}

    phase_map = []
    if "P" in phases:
        phase_map += ["p", "P"]
    if "S" in phases:
        phase_map += ["s", "S"]

    cat = Catalog()
    for (lat, lon, depth), t0 in true_events:
        ev = Event()
        picks_buf = []  # collect, then sort deterministically

        # iterate stations in a deterministic order
        for sta in sorted(stations.keys()):
            sta_lat, sta_lon, elev = stations[sta]
            arrivals = taup_model.get_travel_times_geo(
                source_latitude_in_deg=lat,
                source_longitude_in_deg=lon,
                source_depth_in_km=depth + elev / 1000.0,
                receiver_latitude_in_deg=sta_lat,
                receiver_longitude_in_deg=sta_lon,
                phase_list=phase_map,
            )
            bias = float(station_time_offsets_s.get(sta, 0.0))

            # deterministically include only wanted phases, and sort arrivals
            arr_keep = [a for a in arrivals if a.name in ("p", "s", "P", "S")]
            arr_keep.sort(key=lambda a: (a.name, a.time))

            for a in arr_keep:
                picks_buf.append(
                    Pick(
                        time=t0 + a.time + bias,
                        time_errors=QuantityError(
                            uncertainty=pick_uncertainty, confidence_level=100.0
                        ),
                        phase_hint=a.name,
                        waveform_id=WaveformStreamID(
                            network_code="NET", station_code=sta, channel_code="HHZ"
                        ),
                    )
                )

        # final deterministic order of picks: by (station, phase, time)
        picks_buf.sort(
            key=lambda p: (
                p.waveform_id.station_code or "",
                p.phase_hint or "",
                float(p.time) if hasattr(p.time, "__float__") else p.time.timestamp,
            )
        )
        ev.picks.extend(picks_buf)

        ev.true_location = np.array([lat, lon, depth])  # type: ignore[attr-defined]
        ev.true_origin_time = t0  # type: ignore[attr-defined]
        cat.append(ev)
    return cat


def prepare_geometry_and_biases(builder, kwargs, base_event_loc, base_t, bias, rng):
    stations_nominal = builder(base_event_loc, **kwargs)
    inventory = build_inventory_with_variations(
        stations_nominal, start_date=base_t - 3600, loc_jitter_km=0.0, rng=rng
    )
    stations_used = {
        sta.code: (sta.latitude, sta.longitude, stations_nominal[sta.code][2])
        for sta in inventory[0].stations
    }
    station_time_offsets_s = {
        code: float(rng.choice([-1, 1]) * rng.uniform(0.0, bias))
        for code in stations_used.keys()
    }
    return inventory, stations_used, station_time_offsets_s


# -------------------- Core run utilities --------------------


def locator_factory(mode, engine, phases, model_path):
    from hypy import EventLocator

    if mode == "ray":
        return EventLocator(model_path, phases=phases, mode="ray", engine=engine)
    return EventLocator(
        model_path,
        phases=phases,
        mode="table",
        engine=engine,
        max_distance=250.0,
        distance_step=1.0,
        max_depth=50.0,
        depth_step=2.0,
    )


def effective_time_tolerance(base_tol_s, station_time_offsets_s):
    pad = (
        max(abs(v) for v in station_time_offsets_s.values())
        if station_time_offsets_s
        else 0.0
    )
    return base_tol_s + pad + 0.05


def _assert_errs(true_loc, true_t, solved_loc, solved_t):
    if (
        any(np.isnan(v) for v in solved_loc)
        or solved_t is None
        or (isinstance(solved_t, float) and np.isnan(solved_t))
    ):
        return np.nan, np.nan, np.nan
    arc_m, _, _ = gps2dist_azimuth(
        true_loc[0], true_loc[1], solved_loc[0], solved_loc[1]
    )
    return arc_m / 1000.0, abs(true_loc[2] - solved_loc[2]), abs(true_t - solved_t)


def _normalize_invalids(invalids):
    if invalids is None:
        return []
    try:
        import numpy as _np

        arr = _np.atleast_1d(invalids).ravel().tolist()
        out = []
        for x in arr:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out
    except Exception:
        try:
            return [int(i) for i in invalids]
        except Exception:
            return []


def _results_dir(base, geometry_name, phases, mode, engine, solve_mode, bias) -> Path:
    """
    test_artifacts/<base>/<geometry>/<phases>/<mode>/<engine>/
    (You had solve_mode/bias commented out; keeping your structure.)
    """
    phase_tag = "".join(phases)
    # bias_tag = f"bias-{bias:.2f}s"
    d = (
        Path("test_artifacts") / base / geometry_name / phase_tag / mode / engine
        # / solve_mode
        # / bias_tag
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _filename_stem(
    geometry_name, phases, mode, engine, solve_mode, solver, bias, nst, nev
) -> str:
    return (
        f"{geometry_name}_ph-{''.join(phases)}_{mode}_{engine}_"
        f"{solve_mode}_{solver}_bias-{bias:.2f}s_st{nst}_ev{nev}"
    )


def plot_and_save(
    stations,
    phases,
    geometry_name,
    mode,
    engine,
    solve_mode,
    solver,
    bias,
    to_plot,
    n_success,
    n_total,
    avg_geo,
    avg_depth,
    avg_time,
):
    outdir_plot = _results_dir(
        "plots", geometry_name, phases, mode, engine, solve_mode, bias
    )
    fname_stem = _filename_stem(
        geometry_name,
        phases,
        mode,
        engine,
        solve_mode,
        solver,
        bias,
        len(stations),
        len(to_plot),
    )

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    st_lats = [v[0] for v in stations.values()]
    st_lons = [v[1] for v in stations.values()]
    ax.scatter(st_lons, st_lats, marker="^", s=35, label="Stations", zorder=2)

    true_xy, est_xy, est_err = [], [], []
    est_true_err = []
    for item in to_plot:
        tlat, tlon, _, _, _ = item["true_location"]
        elat, elon, _, dh, _ = item["solved_location"]
        true_xy.append((tlon, tlat))
        if np.isfinite(elat) and np.isfinite(elon):
            est_xy.append((elon, elat))
            est_err.append(dh)
            dist, _, _ = gps2dist_azimuth(tlat, tlon, elat, elon)
            dist3d = np.sqrt(
                (dist / 1000.0) ** 2
                + (item["true_location"][2] - item["solved_location"][2]) ** 2
            )
            est_true_err.append(dist3d)

    if true_xy:
        tlons, tlats = map(np.array, zip(*true_xy))
        ax.scatter(
            tlons, tlats, marker="x", s=60, zorder=4, label="True events", alpha=0.9
        )

    if est_xy:
        elons, elats = map(np.array, zip(*est_xy))
        for x, y, dh in zip(elons, elats, est_err):
            ax.errorbar(
                [x],
                [y],
                xerr=[kilometers2degrees(dh / 1000)],
                yerr=[kilometers2degrees(dh / 1000)],
                fmt="o",
                ecolor="k",
                mec="k",
                ms=3,
                mew=0.8,
                zorder=5,
                alpha=0.6,
            )
        ax.scatter(elons, elats, s=12, zorder=5, label="Estimates", alpha=0.8)

    all_lats = st_lats + [
        v
        for v in [it["true_location"][0] for it in to_plot]
        + [it["solved_location"][0] for it in to_plot]
        if np.isfinite(v)
    ]
    all_lons = st_lons + [
        v
        for v in [it["true_location"][1] for it in to_plot]
        + [it["solved_location"][1] for it in to_plot]
        if np.isfinite(v)
    ]
    ax.set_xlim(min(all_lons) - 0.12, max(all_lons) + 0.12)
    ax.set_ylim(min(all_lats) - 0.12, max(all_lats) + 0.12)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, linestyle=":")
    ax.set_aspect("equal", adjustable="box")

    phase_tag = "".join(phases)
    ax.set_title(
        f"{geometry_name} | phases={phase_tag}\n"
        f"{mode}/{engine} | {solve_mode}/{solver}\n"
        f"{n_success}/{n_total} relocated\n"
        f"avg Geo {avg_geo:.3f} km | Depth {avg_depth:.3f} km | Time {avg_time:.3f} s"
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fontsize=8,
        frameon=True,
        title="Layers",
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 1])

    png_path = outdir_plot / f"{fname_stem}.png"
    fig.savefig(
        png_path,
        dpi=300,
        facecolor="white",
        edgecolor="none",
        orientation="portrait",
        format="png",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.1,
        metadata={
            "Title": f"{geometry_name} relocation results",
            "Author": "Pytest synthetic relocation",
            "Description": (
                f"{n_success}/{n_total} relocated | "
                f"avg Geo {avg_geo:.3f} km | Depth {avg_depth:.3f} km | Time {avg_time:.3f} s"
            ),
        },
    )
    plt.close(fig)
    logger.info(f"Saved plot: {png_path}")

    outdir_results = _results_dir(
        "results", geometry_name, phases, mode, engine, solve_mode, bias
    )

    _save_results_json(
        outdir_results,
        fname_stem,
        geometry_name,
        phases,
        mode,
        engine,
        solve_mode,
        solver,
        bias,
        stations,
        to_plot,
        avg_geo,
        avg_depth,
        avg_time,
        n_success=n_success,
        n_total=n_total,
    )


# -------------------- Caching + scenario execution --------------------


def _seed_all(
    geometry_name, phases, mode, engine, solve_mode, solver, bias, n_stations, n_events
) -> int:
    key = (
        f"{geometry_name}|{''.join(phases)}|{mode}|{engine}|{solve_mode}|{solver}|"
        f"{bias:.6f}|nst{n_stations}|nev{n_events}"
    )
    return zlib.adler32(key.encode("utf-8")) & 0xFFFFFFFF


def _iasp91_tvel_path_local():
    return _iasp91_tvel_path()


def load_or_build_data(scn: Scenario, taup_model) -> DataBundle:
    rng = np.random.default_rng(scn.seed)
    cached = load_cache(
        scn.geometry_name,
        scn.phases,
        scn.mode,
        scn.engine,
        scn.solve_mode,
        scn.solver,
        scn.bias,
        scn.n_stations,
        scn.n_events,
        scn.seed,
    )
    if cached:
        inventory, cached_catalog, TRUE_EVENTS, station_time_offsets_s, paths = cached
        logger.info(f"[CACHE] Loaded {paths['inv'].name} / {paths['cat'].name}")
        net = inventory[0]
        stations_used = {
            sta.code: (sta.latitude, sta.longitude, float(sta.elevation or 0.0))
            for sta in net.stations
        }
        return DataBundle(
            inventory,
            stations_used,
            TRUE_EVENTS,
            cached_catalog,
            station_time_offsets_s,
        )
    logger.warning("[CACHE] No cache found, building synthetic data...")

    TRUE_EVENTS, base_t = make_event_list(
        n_events=scn.n_events, spacing_s=120.0, spread_km=8.0
    )
    builder, kwargs_base, _ = GEOMS[scn.geometry_name]
    kwargs = adjust_geometry_kwargs(builder, kwargs_base, scn.n_stations)
    inventory, stations_used, station_time_offsets_s = prepare_geometry_and_biases(
        builder, kwargs, TRUE_EVENTS[0][0], TRUE_EVENTS[0][1], scn.bias, rng=rng
    )
    cached_catalog = make_catalog_with_biases(
        taup_model,
        stations_used,
        TRUE_EVENTS,
        phases=scn.phases,
        station_time_offsets_s=station_time_offsets_s,
    )
    paths = save_cache(
        inventory,
        cached_catalog,
        TRUE_EVENTS,
        station_time_offsets_s,
        scn.geometry_name,
        scn.phases,
        scn.mode,
        scn.engine,
        scn.solve_mode,
        scn.solver,
        scn.bias,
        scn.n_stations,
        scn.n_events,
        scn.seed,
    )
    logger.info(f"[CACHE] Saved {paths['inv'].name} / {paths['cat'].name}")
    return DataBundle(
        inventory, stations_used, TRUE_EVENTS, cached_catalog, station_time_offsets_s
    )


def run_scenario(scn: Scenario, taup_model, relocation_summary):
    from obspy.core.event import Catalog

    model_path = _iasp91_tvel_path_local()

    def _make_locator():
        return locator_factory(scn.mode, scn.engine, scn.phases, model_path)

    data = load_or_build_data(scn, taup_model)

    # Locate (batch or single)
    if scn.solve_mode == "batch":
        locator = _make_locator()
        locator.load(data.inventory, data.cached_catalog)
        est_events, invalids = locator.locate(
            max_iterations=1000, trial_depth=5.0, solver=scn.solver
        )
        invalids = _normalize_invalids(invalids)
        assert len(est_events) == len(data.true_events)
        true_events = data.true_events
        tags = [f"ev{i + 1}" for i in range(len(true_events))]
    else:
        est_events, true_events, tags, invalids = [], [], [], []
        n = len(data.true_events)
        indices = range(n)
        # indices = range(n - 1, -1, -1)  # reverse order to catch cross-run mutation bugs
        for i in indices:
            per_ev_seed = zlib.adler32(f"{scn.seed}|ev={i}".encode()) & 0xFFFFFFFF
            with deterministic_seed(per_ev_seed):
                logger.debug(f"[RELOCATION] Single event locate ev{i + 1}/{n}")
                logger.debug(
                    f"  True loc: {data.true_events[i][0]}, t0: {data.true_events[i][1]}"
                )
                logger.debug(f"  Tag: ev{i + 1}")
                # deep-copy to avoid cross-run mutation
                single_cat = Catalog(events=[deepcopy(data.cached_catalog[i])])
                locator = _make_locator()
                locator.load(data.inventory, single_cat)
                out, inv = locator.locate(
                    max_iterations=1000, trial_depth=5.0, solver=scn.solver
                )
            assert len(out) == 1
            est_events.append(out[0])
            true_events.append(data.true_events[i])
            tags.append(f"ev{i + 1}")
            if len(_normalize_invalids(inv)) > 0:
                invalids.append(i)

    invalid_idx = set(_normalize_invalids(invalids))
    # invalid_idx = set([50 - 1 - i for i in invalids])  # reverse for single-event mode
    for i, ev in enumerate(est_events):
        if not getattr(ev, "origins", None):
            invalid_idx.add(i)

    total_ev = len(true_events)
    success_ev = total_ev - len(invalid_idx)

    to_plot = []
    sum_geo = sum_depth = sum_time = 0.0
    n_err = 0

    for idx, (e, ((tlat, tlon, tdepth), t0), tag) in enumerate(
        zip(est_events, true_events, tags)
    ):
        if idx in invalid_idx:
            solved_loc = (np.nan, np.nan, np.nan, np.nan, np.nan)
            solved_t = np.nan
        else:
            origin = e.origins[-1]
            solved_loc = (
                origin.latitude,
                origin.longitude,
                origin.depth / 1000.0,
                origin.origin_uncertainty.horizontal_uncertainty,
                origin.depth_errors.uncertainty,
            )
            solved_t = origin.time.timestamp

        true_loc = (tlat, tlon, tdepth, 0.0, 0.0)
        true_t = t0.timestamp
        geo_km, dep_km, time_s = _assert_errs(
            (true_loc[0], true_loc[1], true_loc[2]),
            true_t,
            (solved_loc[0], solved_loc[1], solved_loc[2]),
            solved_t,
        )

        if np.isfinite(geo_km) and np.isfinite(dep_km) and np.isfinite(time_s):
            sum_geo += geo_km
            sum_depth += dep_km
            sum_time += time_s
            n_err += 1

        to_plot.append(
            {
                "true_location": true_loc,
                "solved_location": solved_loc,
                "solved_errs": (geo_km, dep_km, time_s),
                "tag": tag,
            }
        )

    avg_geo = sum_geo / n_err if n_err else float("nan")
    avg_depth = sum_depth / n_err if n_err else float("nan")
    avg_time = sum_time / n_err if n_err else float("nan")

    logger.info(
        f"[RELOCATION] {scn.geometry_name} | ph={''.join(scn.phases)} | "
        f"{scn.mode}/{scn.engine} | {scn.solve_mode}/{scn.solver} | "
        f"bias={scn.bias:.2f}s | nst={scn.n_stations} | nev={scn.n_events} -> "
        f"{success_ev}/{total_ev} relocated | "
        f"avg Geo {avg_geo:.3f} km | Depth {avg_depth:.3f} km | Time {avg_time:.3f} s"
    )

    if relocation_summary is not None:
        relocation_summary["total"] += total_ev
        relocation_summary["success"] += success_ev
        relocation_summary["sum_geo"] += sum_geo
        relocation_summary["sum_depth"] += sum_depth
        relocation_summary["sum_time"] += sum_time
        relocation_summary["n_err"] += n_err

    plot_and_save(
        data.stations_used,
        scn.phases,
        scn.geometry_name,
        scn.mode,
        scn.engine,
        scn.solve_mode,
        scn.solver,
        scn.bias,
        to_plot,
        success_ev,
        total_ev,
        avg_geo,
        avg_depth,
        avg_time,
    )
