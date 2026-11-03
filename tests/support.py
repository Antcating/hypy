"""Shared helpers for the Hypothesis-based fuzz suite.

This module is intentionally *not* a ``test_*`` module so pytest does not try
to collect it.  It provides:

* cached, session-wide velocity models and ray tracers (building numba kernels
  or TauP models is expensive, so we do it once);
* ObsPy inventory/catalog builders that synthesise physically-consistent picks
  from a tracer, so the full location pipeline can be exercised end-to-end.
"""

from __future__ import annotations

import functools
import os
from typing import Sequence, Mapping, Sequence

import numpy as np
import obspy
from obspy import UTCDateTime
from obspy.core.event import Event, Origin, Pick, WaveformStreamID
from obspy.core.event.base import QuantityError
from obspy.core.inventory import Inventory, Network, Site, Station

# --------------------------------------------------------------------------- #
# Velocity models / tracers (built once per session)
# --------------------------------------------------------------------------- #

TVEL_PATH = os.path.join(os.path.dirname(obspy.__file__), "taup", "data", "iasp91.tvel")


@functools.lru_cache(maxsize=None)
def get_tvel_model():
    """Return a cached :class:`TvelModel` built from the bundled iasp91
    model."""
    from hypy.model.layered.tvel import TvelModel

    return TvelModel(TVEL_PATH)


@functools.lru_cache(maxsize=None)
def get_hypo_tracer(max_distance_km: float = 800.0, max_depth_km: float = 50.0):
    """Return a cached fast numba ``HypoRayTracer`` over iasp91."""
    from hypy.propagation.layered.ray_tracer.hypo.tracer import HypoRayTracer

    return HypoRayTracer(
        model=get_tvel_model(),
        max_distance_km=max_distance_km,
        max_depth_km=max_depth_km,
        phases=("p", "s"),
    )


# --------------------------------------------------------------------------- #
# ObsPy data builders
# --------------------------------------------------------------------------- #


def km_to_deg_offsets(lat0_deg: float, d_north_km: float, d_east_km: float):
    """Convert small north/east offsets (km) to (dlat, dlon) degrees."""
    dlat = d_north_km / 111.0
    dlon = d_east_km / (111.0 * np.cos(np.radians(lat0_deg)))
    return dlat, dlon


def ring_stations(
    lat0: float,
    lon0: float,
    radius_km: float,
    n: int,
    *,
    prefix: str = "S",
    network: str = "NW",
) -> dict[str, tuple[float, float, float]]:
    """Return ``{code: (lat, lon, elev_m)}`` for ``n`` stations on a ring."""
    stations: dict[str, tuple[float, float, float]] = {}
    for i, theta in enumerate(np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)):
        north = radius_km * np.cos(theta)
        east = radius_km * np.sin(theta)
        dlat, dlon = km_to_deg_offsets(lat0, north, east)
        stations[f"{prefix}{i:02d}"] = (lat0 + dlat, lon0 + dlon, 0.0)
    return stations


def build_inventory(
    stations: Mapping[str, tuple[float, float, float]],
    network_code: str = "NW",
) -> Inventory:
    """Build an ObsPy :class:`Inventory` from ``{code: (lat, lon, elev)}``."""
    net = Network(code=network_code)
    for code, (lat, lon, elev) in stations.items():
        net.stations.append(
            Station(
                code=code,
                latitude=lat,
                longitude=lon,
                elevation=elev,
                site=Site(name=code),
            )
        )
    return Inventory(networks=[net], source="fuzz")


def build_pick(
    station_code: str,
    phase: str,
    time,
    *,
    uncertainty: float | None = 0.2,
    network_code: str = "NW",
) -> Pick:
    """Build a single ObsPy :class:`Pick`."""
    time_errors = (
        None if uncertainty is None else QuantityError(uncertainty=uncertainty)
    )
    return Pick(
        time=time,
        phase_hint=phase,
        waveform_id=WaveformStreamID(
            network_code=network_code,
            station_code=station_code,
            channel_code="HHZ",
        ),
        time_errors=time_errors,
    )


def build_synthetic_event(
    tracer,
    stations: Mapping[str, tuple[float, float, float]],
    *,
    event_lat: float,
    event_lon: float,
    event_depth_km: float,
    phases: Sequence[str] = ("P", "S"),
    origin_time=None,
    uncertainty: float = 0.1,
    noise_s: float = 0.0,
    rng: np.random.Generator | None = None,
    network_code: str = "NW",
    depth_type: str | None = None,
) -> Event:
    """Synthesise an ObsPy event whose picks match ``tracer`` travel times.

    Picks whose travel time cannot be computed (out of range) are
    skipped, so the returned event may contain fewer picks than
    ``len(stations) * n_phases``.
    """
    if origin_time is None:
        origin_time = UTCDateTime(2024, 1, 1)
    picks: list[Pick] = []
    for code, (lat, lon, elev) in stations.items():
        for phase in phases:
            try:
                _, travel_time, _, _ = tracer.get_arrival(
                    event_lat, event_lon, event_depth_km, lat, lon, elev, phase
                )
            except Exception:
                continue
            if travel_time is None or not np.isfinite(travel_time):
                continue
            perturbation = 0.0
            if noise_s and rng is not None:
                perturbation = float(rng.normal(0.0, noise_s))
            picks.append(
                build_pick(
                    code,
                    phase,
                    origin_time + float(travel_time) + perturbation,
                    uncertainty=uncertainty,
                    network_code=network_code,
                )
            )
    event = Event(picks=picks)
    if depth_type is not None:
        event.origins = [Origin(depth_type=depth_type)]
    return event
