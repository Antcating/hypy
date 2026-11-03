import logging

import numpy as np
import pandas as pd
import pytest
from obspy import UTCDateTime  # type: ignore[import-untyped]
from obspy.core.event import Event  # type: ignore[import-untyped]

from hypy.locator.config import LocatorConfig
from hypy.locator.data import DataHandler
from hypy.utils.weighting import time_error_weight

from ..support import build_inventory, build_pick


def make_handler(
    phases,
    min_picks_per_event,
    time_error_floor_s,
    fix_depth_km=None,
):
    """Build a DataHandler backed by a minimal LocatorConfig for tests."""
    config = LocatorConfig(
        phases=phases,
        min_picks_per_event=min_picks_per_event,
        time_error_floor_s=time_error_floor_s,
        derivative_method="numerical",
        fix_depth_km=fix_depth_km,
    )
    return DataHandler(config=config)


def build_event(picks):
    return Event(picks=picks)


def test_load_data_raises_for_insufficient_picks():
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)
    events = [build_event([build_pick("STA1", "P", t0)])]

    handler = make_handler(
        phases=("P",),
        min_picks_per_event=2,
        time_error_floor_s=0.1,
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
        time_error_floor_s=0.1,
    )

    with caplog.at_level(logging.WARNING, logger="hypy"):
        handler.load_data(inventory, [event])

    assert "Stations present in picks but missing in inventory" in caplog.text
    assert handler.stations == [("NW", "STA1")]
    assert handler.n_stations == 1
    assert handler.stations_locations.shape == (1, 3)


def test_fixed_depth_mask_and_global_fix_depth():
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0), "STA2": (1.0, 1.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)

    ev1 = build_event(
        [
            build_pick("STA1", "P", t0 + 1),
            build_pick("STA2", "P", t0 + 2),
        ]
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
        time_error_floor_s=0.1,
    )
    handler.load_data(inventory, [ev1, ev2], fixed_depth_mask=[True, False])
    assert handler.fixed_depth_mask == [True, False]

    fixed_handler = make_handler(
        phases=("P",),
        min_picks_per_event=1,
        time_error_floor_s=0.1,
        fix_depth_km=5.0,
    )
    fixed_handler.load_data(inventory, [ev1, ev2], fixed_depth_mask=[False, False])
    assert fixed_handler.fixed_depth_mask == [True, True]

    with pytest.raises(ValueError, match="one boolean per event"):
        handler.load_data(inventory, [ev1, ev2], fixed_depth_mask=[True])


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
        time_error_floor_s=0.1,
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
    assert handler.station_time_corrections.shape == (6,)
    np.testing.assert_array_equal(handler.station_time_corrections, np.zeros(6))


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
        time_error_floor_s=0.1,
    )
    with caplog.at_level(logging.DEBUG, logger="hypy"):
        handler.load_data(inventory, [event_invalid, event_valid])

    assert "only 1 picks < 3 required" in caplog.text
    assert bool(handler.invalid_events[0]) is True
    assert bool(handler.invalid_events[1]) is False
    assert np.isnan(handler.arrivals[0]).all()


def test_event_skipped_if_no_matching_phase(caplog, monkeypatch):
    """Unrecognized ``phase_hint`` values are dropped, not fatal."""
    monkeypatch.setattr(logging.getLogger("hypy"), "propagate", True)
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0), "STA2": (0.0, 1.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)
    event_unmatched = build_event(
        [
            build_pick("STA1", "X", t0 + 1),
            build_pick("STA2", "Y", t0 + 2),
        ]
    )
    # A loadable companion event, so the dataset still has usable stations.
    event_valid = build_event(
        [
            build_pick("STA1", "P", t0 + 3),
            build_pick("STA2", "P", t0 + 4),
        ]
    )

    handler = make_handler(
        phases=("P", "S"),
        min_picks_per_event=1,
        time_error_floor_s=0.1,
    )
    with caplog.at_level(logging.WARNING, logger="hypy"):
        handler.load_data(inventory, [event_unmatched, event_valid])

    assert "Skipping pick NW.STA1 'X': phase not in ('p', 's')" in caplog.text
    assert bool(handler.invalid_events[0]) is True
    assert bool(handler.invalid_events[1]) is False
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
        time_error_floor_s=0.1,
    )
    with caplog.at_level(logging.DEBUG, logger="hypy"):
        handler.load_data(inventory, [event_invalid, event_valid])

    assert "only 2 usable station-phase picks < 3 required" in caplog.text
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
        time_error_floor_s=0.2,
    )

    with caplog.at_level(logging.DEBUG, logger="hypy"):
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
        time_error_floor_s=0.1,
    )
    handler.load_data(inventory, events)

    assert handler.events is not events
    assert handler.events[0] is not events[0]


def test_load_data_applies_station_time_corrections(caplog, monkeypatch):
    """``load_data`` fills the correction vector without mutating arrivals."""
    monkeypatch.setattr(logging.getLogger("hypy"), "propagate", True)
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0), "STA2": (1.0, 1.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)
    event = build_event(
        [
            build_pick("STA1", "P", t0 + 10, uncertainty=0.2),
            build_pick("STA1", "S", t0 + 20, uncertainty=0.2),
            build_pick("STA2", "P", t0 + 14, uncertainty=0.2),
            build_pick("STA2", "S", t0 + 24, uncertainty=0.2),
        ]
    )
    handler = make_handler(
        phases=("P", "S"),
        min_picks_per_event=1,
        time_error_floor_s=0.1,
    )
    with caplog.at_level(logging.WARNING, logger="hypy"):
        handler.load_data(
            inventory,
            [event],
            station_time_corrections=pd.DataFrame(
                [
                    {"network": "NW", "station": "STA1", "delay_s": 0.10},
                    {
                        "network": "NW",
                        "station": "STA1",
                        "phase": "S",
                        "delay_s": 0.25,
                    },
                    {"network": "ZZ", "station": "GONE", "delay_s": 0.40},
                ]
            ),
        )

    # Column order is station-major: STA1-P, STA1-S, STA2-P, STA2-S.
    np.testing.assert_allclose(handler.station_time_corrections, [0.10, 0.25, 0.0, 0.0])
    assert "GONE" in caplog.text

    arrivals_before = handler.arrivals.copy()
    handler.theoretical_travel_times[:] = 1.0
    handler.origin_times[:] = 0.0
    residuals = handler.observation_residuals()
    np.testing.assert_allclose(
        residuals[0],
        arrivals_before[0] - 1.0 - handler.station_time_corrections,
    )
    np.testing.assert_allclose(handler.arrivals, arrivals_before)
