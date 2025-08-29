# test_event_locator_pytest.py
import logging
import os
import time
from pathlib import Path

import numpy as np
import pytest
from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Pick, WaveformStreamID
from obspy.core.event.base import QuantityError
from obspy.core.inventory import Inventory, Network, Site, Station
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel

from hypy.event_locator import EventLocator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- Fixtures & helpers ----------


@pytest.fixture(scope="session", autouse=True)
def _seed():
    np.random.seed(42)


@pytest.fixture(scope="session")
def taup_model():
    return TauPyModel(model="iasp91")


@pytest.fixture(scope="session")
def station_coords():
    base = {
        "STA01": (10.005, 10.005, 0),
        "STA02": (10.005, 10.035, 0),
        "STA03": (10.005, 10.065, 0),
        "STA04": (10.005, 10.095, 0),
        "STA05": (10.035, 10.005, 0),
        "STA06": (10.035, 10.035, 0),
        "STA07": (10.035, 10.065, 0),
        "STA08": (10.035, 10.095, 0),
        "STA09": (10.065, 10.005, 0),
        "STA10": (10.065, 10.035, 0),
        "STA11": (10.065, 10.065, 0),
        "STA12": (10.065, 10.095, 0),
        "STA13": (10.095, 10.005, 0),
        "STA14": (10.095, 10.035, 0),
        "STA15": (10.095, 10.065, 0),
        "STA16": (10.095, 10.095, 0),
    }
    extended = dict(base)
    extended.update(
        {
            "STA17": (10.125, 10.005, 0),
            "STA18": (10.125, 10.035, 0),
            "STA19": (10.125, 10.065, 0),
            "STA20": (10.125, 10.095, 0),
        }
    )
    return base, extended


@pytest.fixture(scope="session")
def inventory(station_coords):
    _, stations_extended = station_coords
    net = Network(code="NET", start_date=UTCDateTime("2020-01-01"))
    for sta, (lat, lon, elev) in stations_extended.items():
        net.stations.append(
            Station(
                code=sta,
                latitude=lat,
                longitude=lon,
                elevation=elev,
                start_date=UTCDateTime("2020-01-01"),
                site=Site(name=sta),
            )
        )
    return Inventory(networks=[net], source="synthetic")


@pytest.fixture(scope="session")
def true_event_params():
    # True event (lat, lon, depth[km]) and origin time
    return (10.03, 10.07, 5.0), UTCDateTime("2023-01-01T00:00:00.000Z")


def make_events(
    model,
    station_coords_base,
    phases,
    n_events=1,
    pick_uncertainty=0.01,
    location_variation=0.01,
    depth_variation=0.1,
    time_variation=0.1,
    true_loc=(10.03, 10.07, 5.0),
    true_t=UTCDateTime("2023-01-01T00:00:00.000Z"),
):
    events = Catalog()
    for _ in range(n_events):
        lat = true_loc[0] + np.random.uniform(-location_variation, location_variation)
        lon = true_loc[1] + np.random.uniform(-location_variation, location_variation)
        depth = true_loc[2] + np.random.uniform(-depth_variation, depth_variation)
        origin_time = true_t + np.random.uniform(-time_variation, time_variation)

        event = Event()
        synthetic_phases = []
        if "P" in phases:
            synthetic_phases.append("ttp")
        if "S" in phases:
            synthetic_phases.append("tts")

        for sta, (sta_lat, sta_lon, elev) in station_coords_base.items():
            arrivals = model.get_travel_times_geo(
                source_latitude_in_deg=lat,
                source_longitude_in_deg=lon,
                source_depth_in_km=depth + elev / 1000.0,
                receiver_latitude_in_deg=sta_lat,
                receiver_longitude_in_deg=sta_lon,
                phase_list=synthetic_phases,
            )
            for arr in [a for a in arrivals if a.name in ["p", "s", "P", "S"]]:
                pick_time = origin_time + arr.time
                pick = Pick(
                    time=pick_time
                    + np.random.uniform(-pick_uncertainty / 2, pick_uncertainty / 2),
                    time_errors=QuantityError(
                        uncertainty=pick_uncertainty, confidence_level=100.0
                    ),
                    phase_hint=arr.name,
                    waveform_id=WaveformStreamID(
                        network_code="NET", station_code=sta, channel_code="ENZ"
                    ),
                )
                event.picks.append(pick)

        # stash truth for assertions
        event.true_location = np.array([lat, lon, depth])
        event.true_origin_time = origin_time
        events.append(event)
    return events


def assert_events(locations, origin_times, events):
    true_locations = np.array([e.true_location for e in events])
    true_origin_times = np.array([e.true_origin_time.timestamp for e in events])

    arc_errors_km = []
    depth_errors = []
    for true_loc, loc in zip(true_locations, locations):
        arc_m, _, _ = gps2dist_azimuth(true_loc[0], true_loc[1], loc[0], loc[1])
        depth_error = np.abs(true_loc[2] - loc[2])
        arc_errors_km.append(arc_m / 1000.0)
        depth_errors.append(depth_error)

    time_errors = [
        np.abs(t_true - t_est) for t_true, t_est in zip(true_origin_times, origin_times)
    ]

    arc_errors_km = np.array(arc_errors_km)
    depth_errors = np.array(depth_errors)
    time_errors = np.array(time_errors)

    logger.info(f"Original Locations {true_locations}")
    logger.info(f"Locations {locations}")
    logger.info(f"Depth Errors (km) {float(np.mean(depth_errors))}")
    logger.info(f"Time Errors (s) {float(np.mean(time_errors))}")

    assert np.all(arc_errors_km < 5.0)
    assert np.all(depth_errors < 3.0)
    assert np.all(time_errors < 1.0)
    assert locations.shape == (len(events), 3)


def _iasp91_tvel_path():
    import obspy

    obspy_path = Path(obspy.__file__).parents[0].as_posix()
    return os.path.join(obspy_path, "taup/data/iasp91.tvel")


@pytest.mark.parametrize(
    "phases, engine",
    [
        (["P"], "pyrocko"),
        (["P"], "taup"),
        (["S"], "pyrocko"),
        (["S"], "taup"),
        (["P", "S"], "pyrocko"),
        (["P", "S"], "taup"),
    ],
)
def test_locate_ray(
    taup_model, station_coords, inventory, true_event_params, phases, engine
):
    model_path = _iasp91_tvel_path()
    (true_loc, true_t) = true_event_params
    base, _ = station_coords

    events = make_events(taup_model, base, phases, true_loc=true_loc, true_t=true_t)

    t0 = time.time()
    locator = EventLocator(model_path, phases=phases, mode="ray", engine=engine)
    locator.load_data(inventory, events)
    events = locator.locate(max_iterations=100, trial_depth=5.0)
    elapsed = time.time() - t0
    logger.info("Elapsed time (P picks only): %.3f s", elapsed)

    locations = np.array(
        [
            [
                e.origins[-1].latitude,
                e.origins[-1].longitude,
                e.origins[-1].depth / 1000.0,
            ]
            for e in events
        ]
    )
    origin_times = np.array([e.origins[-1].time.timestamp for e in events])

    assert_events(locations, origin_times, events)


@pytest.mark.parametrize(
    "phases, engine",
    [
        (["P"], "pyrocko"),
        (["P"], "taup"),
        (["S"], "pyrocko"),
        (["S"], "taup"),
        (["P", "S"], "pyrocko"),
        (["P", "S"], "taup"),
    ],
)
def test_locate_table(
    taup_model, station_coords, inventory, true_event_params, phases, engine
):
    model_path = _iasp91_tvel_path()
    (true_loc, true_t) = true_event_params
    base, _ = station_coords

    events = make_events(taup_model, base, phases, true_loc=true_loc, true_t=true_t)

    t0 = time.time()
    locator = EventLocator(
        model_path,
        phases=phases,
        mode="table",
        engine=engine,
        max_distance=100.0,
        distance_step=1.0,
        max_depth=20.0,
        depth_step=1.0,
    )
    locator.load_data(inventory, events)
    events = locator.locate(max_iterations=100, trial_depth=5.0)
    elapsed = time.time() - t0
    logger.info(f"Elapsed time (S picks only): {elapsed:.3f} s")

    locations = np.array(
        [
            [
                e.origins[-1].latitude,
                e.origins[-1].longitude,
                e.origins[-1].depth / 1000.0,
            ]
            for e in events
        ]
    )
    origin_times = np.array([e.origins[-1].time.timestamp for e in events])

    assert_events(locations, origin_times, events)
