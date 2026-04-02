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

from hypy.locator.data import DataHandler
from hypy.utils.weighting import time_error_weight


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


def test_load_data_raises_for_insufficient_picks():
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)
    events = [build_event([build_pick("STA1", "P", t0)])]

    handler = DataHandler(
        phases=("P",),
        min_picks_per_event=2,
        default_time_error=0.1,
        dim=4,
    )

    with pytest.raises(ValueError, match="required picks/event"):
        handler.load_data(inventory, events)


def test_load_data_warns_for_missing_inventory_entry():
    inventory = build_inventory({"STA1": (1.0, 2.0, 3.0)})
    t0 = UTCDateTime(2024, 1, 1)
    event = build_event(
        [
            build_pick("STA1", "P", t0 + 1),
            build_pick("MISS", "P", t0 + 2),
        ]
    )

    handler = DataHandler(
        phases=("P",),
        min_picks_per_event=1,
        default_time_error=0.1,
        dim=4,
    )

    handler.load_data(inventory, [event])

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

    handler = DataHandler(
        phases=("P",),
        min_picks_per_event=1,
        default_time_error=0.1,
        dim=4,
    )
    handler.load_data(inventory, [ev1, ev2])
    assert handler.fixed_depth_mask == [True, False]

    fixed_handler = DataHandler(
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

    handler = DataHandler(
        phases=("P", "S"),
        min_picks_per_event=1,
        default_time_error=0.1,
        dim=4,
    )
    handler.load_data(inventory, [event])

    assert handler.arrivals.shape == (1, 6)
    assert handler.arrival_time_errors.shape == (1, 6)
    assert handler.theoretical_travel_times.shape == (1, 6)
    assert handler.derivative_matrix.shape == (6, 4)
    assert handler.time_error_W.shape == (6,)
    assert handler.distance_W.shape == (6,)
    assert handler.locations.shape == (1, 3)
    assert handler.origin_times.shape == (1,)
    assert handler.time_refs.shape == (1,)
    assert handler.invalid_events.shape == (1,)


def test_event_skipped_due_to_insufficient_picks():
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

    handler = DataHandler(
        phases=("P", "S"),
        min_picks_per_event=3,
        default_time_error=0.1,
        dim=4,
    )
    handler.load_data(inventory, [event_invalid, event_valid])

    assert bool(handler.invalid_events[0]) is True
    assert bool(handler.invalid_events[1]) is False
    assert np.isnan(handler.arrivals[0]).all()


def test_event_skipped_if_no_matching_phase():
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0), "STA2": (0.0, 1.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)
    event = build_event(
        [
            build_pick("STA1", "X", t0 + 1),
            build_pick("STA2", "Y", t0 + 2),
        ]
    )

    handler = DataHandler(
        phases=("P", "S"),
        min_picks_per_event=1,
        default_time_error=0.1,
        dim=4,
    )
    handler.load_data(inventory, [event])

    assert bool(handler.invalid_events[0]) is True
    assert handler.time_refs[0] == 0.0


def test_event_skipped_due_to_insufficient_unique_stations():
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

    handler = DataHandler(
        phases=("P", "S"),
        min_picks_per_event=3,
        default_time_error=0.1,
        dim=4,
    )
    handler.load_data(inventory, [event_invalid, event_valid])

    assert bool(handler.invalid_events[0]) is True
    assert bool(handler.invalid_events[1]) is False


def test_event_handler_manages_time_errors():
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0), "STA2": (1.0, 1.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)
    picks = [
        build_pick("STA1", "P", t0 + 10, uncertainty=0.5),
        build_pick("STA2", "P", t0 + 14, uncertainty=-1.0),
        build_pick("STA1", "S", t0 + 20, uncertainty=0.25),
        build_pick("STA2", "S", t0 + 24, uncertainty=None),
    ]
    event = build_event(picks)

    handler = DataHandler(
        phases=("P", "S"),
        min_picks_per_event=1,
        default_time_error=0.2,
        dim=4,
    )

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


def test_event_deepcopies_on_load():
    inventory = build_inventory({"STA1": (0.0, 0.0, 0.0)})
    t0 = UTCDateTime(2024, 1, 1)
    events = [build_event([build_pick("STA1", "P", t0 + 1)])]

    handler = DataHandler(
        phases=("P",),
        min_picks_per_event=1,
        default_time_error=0.1,
        dim=4,
    )
    handler.load_data(inventory, events)

    assert handler.events is not events
    assert handler.events[0] is not events[0]
