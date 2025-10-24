import json
from pathlib import Path
import numpy as np


def _float_or_none(x):
    try:
        if x is None:
            return None
        xf = float(x)
        return None if np.isnan(xf) else xf
    except Exception:
        return None


def _save_results_json(
    outdir: Path,
    filename_stem: str,
    geometry_name,
    phases,
    mode,
    engine,
    solve_mode,
    solver,
    bias,
    stations_used,
    to_plot,
    avg_geo,
    avg_depth,
    avg_time,
    n_success,
    n_total,
):
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{filename_stem}.json"

    events_payload = []
    for it in to_plot:
        (tlat, tlon, tdepth, _, _) = it["true_location"]
        (elat, elon, edepth, eh, ez) = it["solved_location"]
        (geo_km, dep_km, time_s) = it["solved_errs"]
        events_payload.append(
            {
                "tag": it["tag"],
                "true": {
                    "lat": _float_or_none(tlat),
                    "lon": _float_or_none(tlon),
                    "depth_km": _float_or_none(tdepth),
                },
                "estimate": {
                    "lat": _float_or_none(elat),
                    "lon": _float_or_none(elon),
                    "depth_km": _float_or_none(edepth),
                    "horiz_uncert_km": _float_or_none((eh or np.nan) / 1000.0),
                    "depth_uncert_km": _float_or_none((ez or np.nan) / 1000.0),
                },
                "errors": {
                    "geo_km": _float_or_none(geo_km),
                    "depth_km": _float_or_none(dep_km),
                    "origin_time_s": _float_or_none(time_s),
                },
            }
        )

    payload = {
        "meta": {
            "geometry": geometry_name,
            "phases": list(phases),
            "mode": mode,
            "engine": engine,
            "solve_mode": solve_mode,
            "solver": solver,
            "bias_s": _float_or_none(bias),
            "n_stations": len(stations_used),
            "n_events": len(to_plot),
            "summary": {
                "relocated": int(n_success),
                "total": int(n_total),
                "avg_geo_km": _float_or_none(avg_geo),
                "avg_depth_km": _float_or_none(avg_depth),
                "avg_time_s": _float_or_none(avg_time),
            },
        },
        "events": events_payload,
    }

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
