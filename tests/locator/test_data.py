import logging

import numpy as np
import pytest
from obspy import UTCDateTime  # type: ignore[import-untyped]
from obspy.core.event import (  # type: ignore[import-untyped]
    Event,
    Origin,
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

from hypy.locator.config import LocatorConfig
from hypy.locator.data import DataHandler
from hypy.utils.weighting import time_error_weight


def make_handler(
    phases,
    min_picks_per_event,
    default_time_error,
    dim=4,
    fix_depth_km=None,
):
    """Build a DataHandler backed by a minimal LocatorConfig for tests."""
    config = LocatorConfig(
        phases=phases,
        min_picks_per_event=min_picks_per_event,
        default_time_error=default_time_error,
        derivative_method="numerical",
        fix_depth_km=fix_depth_km,
        precalculate_velocities=False,
        dim=dim,
    )
    return DataHandler(config=config)


def build_inventory(stations):
    net = Network(code="NW")
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
    return Inventory(networks=[net], source="test")


def build_pick(station_code, phase, t, uncertainty=0.2, network_code="NW"):
    return Pick(
        time=t,
        phase_hint=phase,
        waveform_id=WaveformStreamID(
            network_code=network_code,
            station_code=station_code,
            channel_code="HHZ",
        ),
        time_errors=QuantityError(uncertainty=uncertainty),
    )


def build_event(picks, origin_depth_type=None):
    event = Event(picks=picks)
    if origin_depth_type is not None:
        event.origins = [Origin(depth_type=origin_depth_type)]
    return event


@pytest.mark.parametrize(
    "attr",
    [
        "fix_depth_km",
        "min_picks_per_event",
        "default_time_error",
        "dim",
        "phases",
        "n_phases",
    ],
)
def test_data_handler_raises_for_missing_config(attr):
    handler = DataHandler(config=None)
    with pytest.raises(ValueError, match="Config is not loaded yet."):
        getattr(handler, attr)


def test_data_handler_validates_config():
    config = LocatorConfig(
        phases=("P",),
        min_picks_per_event=2,
        default_time_error=0.1,
        derivative_method="numerical",
        fix_depth_km=None,
        precalculate_velocities=False,
    )
    handler = DataHandler(config=config)
    assert handler.config is config
    assert handler.fix_depth_km is None
    assert handler.min_picks_per_event == 2
    assert handler.default_time_error == 0.1
    assert handler.dim == 4
    assert handler.phases == ("p",)
    assert handler.n_phases == 1


def test_data_handler_lowercases_phases():
    config = LocatorConfig(
        phases=("P", "S"),
        min_picks_per_event=2,
        default_time_error=0.1,
        derivative_method="numerical",
        fix_depth_km=None,
        precalculate_velocities=False,
    )
    handler = DataHandler(config=config)
    assert handler.phases == ("p", "s")
    assert handler.n_phases == 2


def test_load_data_raises_for_insufficient_picks():
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)
    events = [build_event([build_pick("STA1", "P", t0)])]

    handler = make_handler(
        phases=("P",),
        min_picks_per_event=2,
        default_time_error=0.1,
        dim=4,
    )

    with pytest.raises(ValueError, match="required picks/event"):
        handler.load_data(inventory, events)


def test_load_data_warns_for_missing_inventory_entry(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("hypy"), "propagate", True)
    inventory = build_inventory({"STA1": (1.0, 2.0, 3.0)})
    t0 = UTCDateTime(2024, 1, 1)
    event = build_event(
        [
            build_pick("STA1", "P", t0 + 1),
            build_pick("MISS", "P", t0 + 2),
        ]
    )

    handler = make_handler(
        phases=("P",),
        min_picks_per_event=1,
        default_time_error=0.1,
        dim=4,
    )

    with caplog.at_level(logging.WARNING, logger="hypy"):
        handler.load_data(inventory, [event])

    assert "Stations present in picks but missing in inventory" in caplog.text
    assert handler.stations == [("NW", "STA1")]
    assert handler.n_stations == 1
    assert handler.stations_locations.shape == (1, 3)


def test_fixed_depth_mask_respects_global_fix_depth():
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0), "STA2": (1.0, 1.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)

    ev1 = build_event(
        [
            build_pick("STA1", "P", t0 + 1),
            build_pick("STA2", "P", t0 + 2),
        ],
        origin_depth_type="operator assigned",
    )
    ev2 = build_event(
        [
            build_pick("STA1", "P", t0 + 3),
            build_pick("STA2", "P", t0 + 4),
        ]
    )

    handler = make_handler(
        phases=("P",),
        min_picks_per_event=1,
        default_time_error=0.1,
        dim=4,
    )
    handler.load_data(inventory, [ev1, ev2])
    assert handler.fixed_depth_mask == [True, False]

    fixed_handler = make_handler(
        phases=("P",),
        min_picks_per_event=1,
        default_time_error=0.1,
        dim=4,
        fix_depth_km=5.0,
    )
    fixed_handler.load_data(inventory, [ev1, ev2])
    assert fixed_handler.fixed_depth_mask == [True, True]


def test_load_event_data_shapes():
    inventory = build_inventory(
        {"STA1": (0.0, 0.0, 0.0), "STA2": (0.0, 1.0, 0.0), "STA3": (1.0, 0.0, 0.0)}
    )
    t0 = UTCDateTime(2024, 1, 1)
    event = build_event(
        [
            build_pick("STA1", "P", t0 + 1),
            build_pick("STA2", "P", t0 + 2),
            build_pick("STA3", "P", t0 + 3),
        ]
    )

    handler = make_handler(
        phases=("P", "S"),
        min_picks_per_event=1,
        default_time_error=0.1,
        dim=4,
    )
    handler.load_data(inventory, [event])

    assert handler.arrivals.shape == (1, 6)
    assert handler.arrival_time_errors.shape == (1, 6)
    assert handler.theoretical_travel_times.shape == (1, 6)
    assert handler.jacobian_matrix.shape == (6, 4)
    assert handler.time_error_W.shape == (6,)
    assert handler.distance_W.shape == (6,)
    assert handler.locations.shape == (1, 3)
    assert handler.origin_times.shape == (1,)
    assert handler.time_refs.shape == (1,)
    assert handler.invalid_events.shape == (1,)


def test_event_skipped_due_to_insufficient_picks(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("hypy"), "propagate", True)
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0), "STA2": (0.0, 1.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)
    event_invalid = build_event([build_pick("STA1", "P", t0 + 1)])
    event_valid = build_event(
        [
            build_pick("STA1", "P", t0 + 2),
            build_pick("STA2", "P", t0 + 3),
            build_pick("STA2", "S", t0 + 4),
        ]
    )

    handler = make_handler(
        phases=("P", "S"),
        min_picks_per_event=3,
        default_time_error=0.1,
        dim=4,
    )
    with caplog.at_level(logging.WARNING, logger="hypy"):
        handler.load_data(inventory, [event_invalid, event_valid])

    assert "only 1 picks < 3 required" in caplog.text
    assert bool(handler.invalid_events[0]) is True
    assert bool(handler.invalid_events[1]) is False
    assert np.isnan(handler.arrivals[0]).all()


def test_event_skipped_if_no_matching_phase(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("hypy"), "propagate", True)
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0), "STA2": (0.0, 1.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)
    event = build_event(
        [
            build_pick("STA1", "X", t0 + 1),
            build_pick("STA2", "Y", t0 + 2),
        ]
    )

    handler = make_handler(
        phases=("P", "S"),
        min_picks_per_event=1,
        default_time_error=0.1,
        dim=4,
    )
    with caplog.at_level(logging.WARNING, logger="hypy"):
        handler.load_data(inventory, [event])

    assert "Skipping pick NW.STA1 X: phase not in ('p', 's')" in caplog.text
    assert bool(handler.invalid_events[0]) is True
    assert handler.time_refs[0] == 0.0


def test_event_skipped_due_to_insufficient_unique_stations(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("hypy"), "propagate", True)
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0), "STA2": (0.0, 1.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)
    event_invalid = build_event(
        [
            build_pick("STA1", "P", t0 + 1),
            build_pick("STA1", "S", t0 + 2),
            build_pick("STA1", "P", t0 + 2.5),
        ]
    )
    event_valid = build_event(
        [
            build_pick("STA1", "P", t0 + 3),
            build_pick("STA2", "P", t0 + 4),
            build_pick("STA2", "S", t0 + 5),
        ]
    )

    handler = make_handler(
        phases=("P", "S"),
        min_picks_per_event=3,
        default_time_error=0.1,
        dim=4,
    )
    with caplog.at_level(logging.WARNING, logger="hypy"):
        handler.load_data(inventory, [event_invalid, event_valid])

    assert "only 1 stations with picks." in caplog.text
    assert bool(handler.invalid_events[0]) is True
    assert bool(handler.invalid_events[1]) is False


def test_event_handler_manages_time_errors(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("hypy"), "propagate", True)
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0), "STA2": (1.0, 1.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)
    picks = [
        build_pick("STA1", "P", t0 + 10, uncertainty=0.5),
        build_pick("STA2", "P", t0 + 14, uncertainty=-1.0),
        build_pick("STA1", "S", t0 + 20, uncertainty=0.25),
        build_pick("STA2", "S", t0 + 24, uncertainty=None),
    ]
    event = build_event(picks)

    handler = make_handler(
        phases=("P", "S"),
        min_picks_per_event=1,
        default_time_error=0.2,
        dim=4,
    )

    with caplog.at_level(logging.WARNING, logger="hypy"):
        handler.load_data(inventory, [event])

    assert bool(handler.invalid_events[0]) is False

    expected_t_ref = np.mean([float(pk.time) for pk in picks])
    assert np.isclose(handler.time_refs[0], expected_t_ref)

    # STA1-P, STA1-S, STA2-P, STA2-S with phase ordering (P, S)
    arr = handler.arrivals[0]
    assert np.isclose(arr[0], float(t0 + 10) - expected_t_ref)
    assert np.isclose(arr[1], float(t0 + 20) - expected_t_ref)
    assert np.isclose(arr[2], float(t0 + 14) - expected_t_ref)
    assert np.isclose(arr[3], float(t0 + 24) - expected_t_ref)

    errs = handler.arrival_time_errors[0]
    assert np.isclose(errs[0], 0.5)
    assert np.isclose(errs[1], 0.25)
    assert np.isclose(errs[2], 0.2)
    assert np.isclose(errs[3], 0.2)

    assert np.isclose(handler.time_error_W[0], time_error_weight(0.5))
    assert np.isclose(handler.time_error_W[1], time_error_weight(0.25))
    assert np.isclose(handler.time_error_W[2], time_error_weight(0.2))
    assert np.isclose(handler.time_error_W[3], time_error_weight(0.2))

    assert "Pick NW.STA2 p: invalid time error. Using floor 0.2 s." in caplog.text
    assert "Pick NW.STA2 s: invalid time error. Using floor 0.2 s." in caplog.text


def test_event_deepcopies_on_load():
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)
    events = [build_event([build_pick("STA1", "P", t0 + 1)])]

    handler = make_handler(
        phases=("P",),
        min_picks_per_event=1,
        default_time_error=0.1,
        dim=4,
    )
    handler.load_data(inventory, events)

    assert handler.events is not events
    assert handler.events[0] is not events[0]
