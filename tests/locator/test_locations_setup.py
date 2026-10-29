"""Tests for heuristic epicenter seeding."""

from types import SimpleNamespace

import numpy as np
import pytest
from obspy import UTCDateTime
from obspy.core.event import Event, Origin, Pick, WaveformStreamID
from obspy.core.event.base import QuantityError
from obspy.core.inventory import Inventory, Network, Site, Station
from obspy.geodetics import degrees2kilometers, kilometer2degrees

from hypy.locator import EventLocator
from hypy.locator.config import RunConfig
from hypy.locator.locations_setup import (
    _line_frame,
    _weighted_epicenter,
    seed_locations,
)


def _pick(station, time, network="NW", phase="P"):
    return SimpleNamespace(
        phase_hint=phase,
        time=time,
        waveform_id=SimpleNamespace(network_code=network, station_code=station),
    )


def _event_stations(stations):
    return {
        (net, sta): (idx, lat, lon, 0.0)
        for idx, (net, sta, lat, lon) in enumerate(stations)
    }


def test_weighted_epicenter_requires_stations():
    """An empty station map cannot produce an epicenter."""
    with pytest.raises(ValueError, match="without stations"):
        _weighted_epicenter([], {})


def test_weighted_epicenter_few_p_stations_stays_at_centroid():
    """Fewer than two P stations keep the weighted point on the mean."""
    stations = [
        ("NW", "W", 0.0, 0.0),
        ("NW", "E", 0.0, 1.0),
    ]
    mean_lat, mean_lon, weighted_lat, weighted_lon, radius_km = _weighted_epicenter(
        [_pick("W", 0.0), _pick("E", 0.0, phase="S")],
        _event_stations(stations),
    )
    assert mean_lat == pytest.approx(0.0)
    assert mean_lon == pytest.approx(0.5)
    assert weighted_lat == pytest.approx(mean_lat)
    assert weighted_lon == pytest.approx(mean_lon)
    assert radius_km >= 1.0


def test_weighted_epicenter_interior_stays_near_center():
    """Equal P times keep both centroids at the network center."""
    stations = [
        ("NW", "N", 0.5, 0.0),
        ("NW", "S", -0.5, 0.0),
        ("NW", "E", 0.0, 0.5),
        ("NW", "W", 0.0, -0.5),
    ]
    picks = [
        _pick("N", 1.0),
        _pick("S", 1.0),
        _pick("E", 1.0),
        _pick("W", 1.0),
    ]
    mean_lat, mean_lon, weighted_lat, weighted_lon, radius_km = _weighted_epicenter(
        picks,
        _event_stations(stations),
    )
    assert mean_lat == pytest.approx(0.0, abs=1e-9)
    assert mean_lon == pytest.approx(0.0, abs=1e-9)
    assert weighted_lat == pytest.approx(0.0, abs=1e-9)
    assert weighted_lon == pytest.approx(0.0, abs=1e-9)
    assert radius_km >= 1.0


def test_weighted_epicenter_one_sided_extrapolates_past_early_centroid():
    """One-sided P arrivals seed beyond the station hull, not on the early centroid."""
    stations = [
        ("NW", "W", 0.0, 0.0),
        ("NW", "C", 0.0, 0.5),
        ("NW", "E", 0.0, 1.0),
    ]
    picks = [
        _pick("W", 0.0),
        _pick("C", 2.0),
        _pick("E", 4.0),
    ]
    mean_lat, mean_lon, weighted_lat, weighted_lon, _radius_km = _weighted_epicenter(
        picks,
        _event_stations(stations),
    )
    assert mean_lat == pytest.approx(0.0, abs=1e-9)
    assert mean_lon == pytest.approx(0.5, abs=1e-9)
    assert weighted_lat == pytest.approx(0.0, abs=1e-9)
    dt = np.asarray([0.0, 2.0, 4.0])
    time_scale_s = max(float(np.median(dt[dt > 0.0])), 0.5)
    weights = np.exp(-dt / time_scale_s)
    weights /= weights.sum()
    early_lon = float(np.sum(weights * np.asarray([0.0, 0.5, 1.0])))
    factor = float(
        np.clip(1.25 + 0.25 * (float(np.max(dt)) / time_scale_s), 1.25, 2.25)
    )
    assert weighted_lon == pytest.approx(mean_lon + factor * (early_lon - mean_lon))
    assert weighted_lon < early_lon < mean_lon


class _DummyPropagationEngine:
    feature_flags = frozenset({"derivative.numerical", "propagation.layered"})
    phases = ("p", "s")
    max_distance_km = 800.0
    max_depth_km = 50.0
    depth_step_km = None
    distance_step_km = None
    name = "dummy"
    velocity_model = SimpleNamespace(model_name="dummy", radius_of_planet_km=6371.0)


class _AbsLongitudeEngine(_DummyPropagationEngine):
    """Travel time is 10 s per degree of source-station longitude."""

    def get_arrival(
        self,
        source_latitude,
        source_longitude,
        source_depth_km,
        receiver_latitude,
        receiver_longitude,
        receiver_elevation_m,
        phase,
        max_distance_km=None,
    ):
        travel_time = abs(float(source_longitude) - float(receiver_longitude)) * 10.0
        return 0.0, travel_time, 0.0, 0.0


def test_heuristic_scores_candidates_with_inversion_engine():
    """Heuristic multi-start keeps the candidate the inversion model fits."""
    t0 = UTCDateTime(2024, 1, 1)
    preferred_lon = 19.0
    stations = [("W", 10.0, 20.0), ("C", 10.0, 20.2), ("E", 10.0, 20.4)]
    net = Network(code="NW")
    for code, lat, lon in stations:
        net.stations.append(
            Station(
                code=code,
                latitude=lat,
                longitude=lon,
                elevation=0.0,
                site=Site(name=code),
            )
        )
    inventory = Inventory(networks=[net], source="test")

    event = Event()
    for code, _lat, lon in stations:
        tt = abs(preferred_lon - lon) * 10.0
        for phase in ("P", "S"):
            event.picks.append(
                Pick(
                    time=t0 + tt,
                    phase_hint=phase,
                    waveform_id=WaveformStreamID(network_code="NW", station_code=code),
                    time_errors=QuantityError(uncertainty=0.1),
                )
            )

    locator = EventLocator(
        propagation_engine=_DummyPropagationEngine(),
        phases=("p", "s"),
        min_picks_per_event=4,
        derivative_method="numerical",
    )
    locator.load_data(inventory, [event])
    kwargs = dict(
        config=locator.config,
        data=locator.data,
        trial_depth_km=10.0,
        depth_step_km=1.0,
        max_depth_km=50.0,
        propagation_engine=None,
    )
    seed_locations(run_config=RunConfig(seed=1, num_trials=1), **kwargs)
    mean_lon = float(locator.data.locations[0, 1])
    assert mean_lon == pytest.approx(20.2, abs=1e-6)

    seed_locations(
        run_config=RunConfig(seed=1, num_trials=2),
        scoring_engine=_AbsLongitudeEngine(),
        **kwargs,
    )
    scored_lon = float(locator.data.locations[0, 1])
    assert scored_lon < mean_lon


def _bulge_inventory_and_event(source_lat=None, origin=None):
    """Three-station line with the middle site 2 km north. Middle P is earliest."""
    t0 = UTCDateTime(2024, 1, 1)
    stations = [("W", 10.0, 20.0), ("C", 10.02, 20.5), ("E", 10.0, 21.0)]
    net = Network(code="NW")
    for code, lat, lon in stations:
        net.stations.append(
            Station(
                code=code,
                latitude=lat,
                longitude=lon,
                elevation=0.0,
                site=Site(name=code),
            )
        )
    event = Event()
    for code, lat, lon in stations:
        if source_lat is None:
            delay = 0.0 if code == "C" else 5.0
        else:
            delay = (
                float(
                    np.hypot(
                        degrees2kilometers(source_lat - lat),
                        degrees2kilometers(20.5 - lon)
                        * max(abs(np.cos(np.radians(lat))), 1e-6),
                    )
                )
                / 6.0
            )
        for phase, scale in (("P", 1.0), ("S", 1.75)):
            event.picks.append(
                Pick(
                    time=t0 + delay * scale,
                    phase_hint=phase,
                    waveform_id=WaveformStreamID(network_code="NW", station_code=code),
                    time_errors=QuantityError(uncertainty=0.1),
                )
            )
    if origin is not None:
        event.origins = [origin]
    locator = EventLocator(
        propagation_engine=_DummyPropagationEngine(),
        phases=("p", "s"),
        min_picks_per_event=4,
        derivative_method="numerical",
    )
    locator.load_data(Inventory(networks=[net], source="test"), [event])
    return locator


class _DistanceEngine(_DummyPropagationEngine):
    """Travel time is epicentral distance over 6 km/s, times 1.75 for S."""

    def get_arrival(
        self,
        source_latitude,
        source_longitude,
        source_depth_km,
        receiver_latitude,
        receiver_longitude,
        receiver_elevation_m,
        phase,
        max_distance_km=None,
    ):
        cos_lat = max(abs(np.cos(np.radians(float(source_latitude)))), 1e-6)
        dist_km = float(
            np.hypot(
                degrees2kilometers(float(source_latitude) - float(receiver_latitude)),
                degrees2kilometers(float(source_longitude) - float(receiver_longitude))
                * cos_lat,
            )
        )
        travel_time = dist_km / 6.0
        if str(phase).lower().startswith("s"):
            travel_time *= 1.75
        return dist_km, travel_time, 0.0, 0.0


def test_line_frame_ignores_a_crossing_network():
    """A network with extent in both directions is not treated as a line."""
    stations = [
        ("NW", "N", 0.5, 0.0),
        ("NW", "S", -0.5, 0.0),
        ("NW", "E", 0.0, 0.5),
        ("NW", "W", 0.0, -0.5),
    ]
    assert _line_frame(_event_stations(stations)) is None


def test_line_frame_of_an_east_west_array_points_east():
    """A long east-west array is a line; its along-axis points east."""
    stations = [
        ("NW", "W", 10.0, 20.0),
        ("NW", "C", 10.02, 20.5),
        ("NW", "E", 10.0, 21.0),
    ]
    axes = _line_frame(_event_stations(stations))
    assert axes is not None
    along_e, along_n, cross_e, cross_n, sigma_cross = axes
    assert along_e == pytest.approx(1.0, abs=0.05)
    assert abs(along_n) < 0.05
    assert cross_n > 0.9
    assert sigma_cross < 5.0


def test_linear_array_single_trial_stays_at_the_network_mean():
    """``num_trials=1`` does not nudge a linear array toward the station bulge."""
    locator = _bulge_inventory_and_event()
    seed_locations(
        config=locator.config,
        data=locator.data,
        run_config=RunConfig(seed=1, num_trials=1),
        trial_depth_km=10.0,
        depth_step_km=1.0,
        max_depth_km=50.0,
        propagation_engine=None,
    )
    center_lat = float(np.mean([10.0, 10.02, 10.0]))
    assert float(locator.data.locations[0, 0]) == pytest.approx(center_lat)
    assert float(locator.data.locations[0, 1]) == pytest.approx(20.5)


def test_linear_array_stays_in_the_strip_within_num_trials():
    """λ_min/λ_max < 0.1 keeps trials in the strip and scores num_trials times."""
    stations = [
        ("NW", "W", 10.0, 20.0),
        ("NW", "C", 10.02, 20.5),
        ("NW", "E", 10.0, 21.0),
    ]
    axes = _line_frame(_event_stations(stations))
    assert axes is not None
    sigma_cross = axes[4]
    _, _, _, _, radius_km = _weighted_epicenter(
        [_pick("W", 0.0), _pick("C", 0.0), _pick("E", 0.0)],
        _event_stations(stations),
    )
    assert sigma_cross < 0.1 * radius_km
    center_lat = float(np.mean([10.0, 10.02, 10.0]))
    calls = {"n": 0}
    engine = _DistanceEngine()
    real_get_arrival = engine.get_arrival

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real_get_arrival(*args, **kwargs)

    engine.get_arrival = _counting
    n_trials = 4
    locator = _bulge_inventory_and_event(
        source_lat=center_lat + float(kilometer2degrees(radius_km))
    )
    seed_locations(
        config=locator.config,
        data=locator.data,
        run_config=RunConfig(seed=1, num_trials=n_trials),
        trial_depth_km=10.0,
        depth_step_km=1.0,
        max_depth_km=50.0,
        propagation_engine=None,
        scoring_engine=engine,
    )
    north_km = abs(degrees2kilometers(float(locator.data.locations[0, 0]) - center_lat))
    assert north_km < radius_km / 2
    assert calls["n"] == n_trials * 6


def test_linear_array_keeps_a_catalog_origin_on_its_side():
    """``num_trials=1`` with ``use_origins`` does not reflect the catalog origin."""
    t0 = UTCDateTime(2024, 1, 1)
    origin_lat = 9.7
    locator = _bulge_inventory_and_event(
        source_lat=10.3,
        origin=Origin(latitude=origin_lat, longitude=20.5, depth=10_000.0, time=t0),
    )
    seed_locations(
        config=locator.config,
        data=locator.data,
        run_config=RunConfig(use_origins=True, seed=1, num_trials=1),
        trial_depth_km=10.0,
        depth_step_km=1.0,
        max_depth_km=50.0,
        propagation_engine=None,
        scoring_engine=_DistanceEngine(),
    )
    assert float(locator.data.locations[0, 0]) == pytest.approx(origin_lat)
