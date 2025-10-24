import json
import hashlib
from pathlib import Path
from obspy import read_inventory, read_events, UTCDateTime  # type: ignore[import-untyped]


def _cache_dir() -> Path:
    d = Path("test_artifacts") / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(
    geometry_name,
    phases,
    mode,
    engine,
    solve_mode,
    solver,
    bias,
    n_stations,
    n_events,
    seed,
):
    key_str = (
        f"{geometry_name}|{','.join(phases)}|{mode}|{engine}|{solve_mode}|{solver}|"
        f"bias={bias:.6f}|nst={n_stations}|nev={n_events}|seed={seed}"
    )
    h = hashlib.sha1(key_str.encode("utf-8")).hexdigest()[:12]
    prefix = (
        f"{geometry_name}_ph-{''.join(phases)}_{mode}_{engine}_"
        f"{solve_mode}_{solver}_bias-{bias:.2f}_st{n_stations}_ev{n_events}"
    )
    return prefix, h


def _cache_paths(prefix, h):
    base = _cache_dir() / f"{prefix}_{h}"
    return {
        "inv": Path(base.as_posix() + ".inventory.xml"),
        "cat": Path(base.as_posix() + ".catalog.xml"),
        "meta": Path(base.as_posix() + ".meta.json"),
    }


def save_cache(
    inventory,
    catalog,
    true_events,
    station_time_offsets_s,
    geometry_name,
    phases,
    mode,
    engine,
    solve_mode,
    solver,
    bias,
    n_stations,
    n_events,
    seed,
):
    prefix, h = _cache_key(
        geometry_name,
        phases,
        mode,
        engine,
        solve_mode,
        solver,
        bias,
        n_stations,
        n_events,
        seed,
    )
    paths = _cache_paths(prefix, h)

    inventory.write(str(paths["inv"]), format="STATIONXML")
    catalog.write(str(paths["cat"]), format="QUAKEML")

    meta = {
        "geometry_name": geometry_name,
        "phases": phases,
        "mode": mode,
        "engine": engine,
        "solve_mode": solve_mode,
        "solver": solver,
        "bias": bias,
        "n_stations": n_stations,
        "n_events": n_events,
        "seed": seed,
        "true_events": [
            {
                "lat": float(lat),
                "lon": float(lon),
                "depth_km": float(depth),
                "origin_time": t.isoformat(),
            }
            for ((lat, lon, depth), t) in true_events
        ],
        "station_time_offsets_s": station_time_offsets_s,
    }
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return paths


def load_cache(
    geometry_name,
    phases,
    mode,
    engine,
    solve_mode,
    solver,
    bias,
    n_stations,
    n_events,
    seed,
):
    prefix, h = _cache_key(
        geometry_name,
        phases,
        mode,
        engine,
        solve_mode,
        solver,
        bias,
        n_stations,
        n_events,
        seed,
    )
    h += "10"
    paths = _cache_paths(prefix, h)
    if not (paths["inv"].exists() and paths["cat"].exists() and paths["meta"].exists()):
        return None

    inventory = read_inventory(str(paths["inv"]))
    catalog = read_events(str(paths["cat"]))
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    true_events = [
        ((e["lat"], e["lon"], e["depth_km"]), UTCDateTime(e["origin_time"]))
        for e in meta["true_events"]
    ]
    station_time_offsets_s = {
        k: float(v) for k, v in meta["station_time_offsets_s"].items()
    }
    return inventory, catalog, true_events, station_time_offsets_s, paths
