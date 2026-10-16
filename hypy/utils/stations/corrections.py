"""Apply known station time-delay corrections to the observation axis.

A correction is a fixed observation term, not an inversion unknown.
Positive ``delay_s`` means the observed arrival is late (clock behind /
delayed site). Residual:

    r = (t_obs - t0) - T_pred - dt_sta
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from hypy.logger import logger
from hypy.utils.seismic import normalize_phase

_REQUIRED = ("network", "station", "delay_s")


def _phase(value) -> Optional[str]:
    """Return ``"p"``/``"s"``, or ``None`` for a station-wide row."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in {"", "*"}:
        return None
    return normalize_phase(text)


def apply_station_time_corrections(
    table: Optional[pd.DataFrame],
    stations: Sequence[tuple[str, str]],
    phases: Sequence[str],
) -> np.ndarray:
    """Broadcast a delay table onto the station-phase observation axis.

    Parameters
    ----------
    table
        ``None`` or a DataFrame with ``network``, ``station``, ``delay_s``
        and optional ``phase``. Missing / NaN / ``*`` phase is station-wide.
        Extra columns are ignored.
    stations
        ``(network, station)`` keys aligned with the locator inventory.
    phases
        Configured phase names. Normalized with
        :func:`hypy.utils.seismic.normalize_phase`.

    Returns
    -------
    numpy.ndarray
        Length ``len(stations) * len(phases)``, station-major.

    Notes
    -----
    Station-wide rows apply first; phase-specific rows override them.
    Unknown stations and phases are ignored with a warning so a master
    network table can be reused across catalogs.
    """
    n_sta, n_pha = len(stations), len(phases)
    out = np.zeros(n_sta * n_pha, dtype=float)
    if table is None:
        return out
    if not isinstance(table, pd.DataFrame):
        raise ValueError(
            "station_time_corrections must be None or a pandas DataFrame "
            "with columns network, station, delay_s."
        )
    missing = [name for name in _REQUIRED if name not in table.columns]
    if missing:
        raise ValueError(
            "Station time-correction table is missing required columns: "
            f"{', '.join(missing)}."
        )

    delays = pd.to_numeric(table["delay_s"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(delays).all():
        raise ValueError("delay_s must be a finite float.")

    networks = ["" if pd.isna(v) else str(v).strip() for v in table["network"]]
    codes = ["" if pd.isna(v) else str(v).strip() for v in table["station"]]
    if not all(networks) or not all(codes):
        raise ValueError(
            "Station time-correction rows must include non-empty "
            "network and station codes."
        )

    row_phases = (
        [_phase(v) for v in table["phase"]]
        if "phase" in table.columns
        else [None] * len(table)
    )
    sta_index = {key: idx for idx, key in enumerate(stations)}
    pha_index = {normalize_phase(phase): idx for idx, phase in enumerate(phases)}
    unknown_stations: set[tuple[str, str]] = set()
    unknown_phases: set[tuple[str, str, str]] = set()

    # Station-wide rows first; stable so later rows still override earlier ones.
    rows = sorted(
        zip(networks, codes, row_phases, delays),
        key=lambda row: row[2] is not None,
    )
    for network, station, phase, delay in rows:
        key = (network, station)
        sta_idx = sta_index.get(key)
        if sta_idx is None:
            unknown_stations.add(key)
            continue
        if phase is None:
            start = n_pha * sta_idx
            out[start : start + n_pha] = delay
            continue
        pha_idx = pha_index.get(phase)
        if pha_idx is None:
            unknown_phases.add((network, station, phase))
            continue
        out[n_pha * sta_idx + pha_idx] = delay

    if unknown_stations:
        logger.warning(
            "Station time corrections ignored for stations not in the loaded "
            "inventory/picks: %s",
            sorted(unknown_stations),
        )
    if unknown_phases:
        logger.warning(
            "Station time corrections ignored for phases not in %s: %s",
            tuple(pha_index),
            sorted(unknown_phases),
        )
    n_applied = np.count_nonzero(out)
    if n_applied:
        logger.info(
            "Applied station time corrections to %s station-phase term(s).",
            n_applied,
        )
    return out
