from pathlib import Path

import numpy as np
from obspy.core.event import Catalog, Event

from hypy.locator.export import (
    OriginExporter,
    events_to_catalog,
    save_catalog,
    save_events,
)
from hypy.locator.types import ErrorResults


def test_events_to_catalog_list():
    ev = Event()
    catalog = events_to_catalog([ev])
    assert isinstance(catalog, Catalog)
    assert len(catalog.events) == 1


def test_events_to_catalog_catalog():
    src = Catalog(events=[Event()])
    result = events_to_catalog(src)
    assert result is src


def test_save_catalog_and_events(tmp_path: Path):
    catalog = Catalog(events=[Event()])
    out1 = tmp_path / "test_quakeml.xml"
    save_catalog(catalog, out1, format="QUAKEML")
    assert out1.exists()

    out2 = tmp_path / "test_events.xml"
    save_events([Event()], out2, format="QUAKEML")
    assert out2.exists()


def build_test_event_data(validity):
    data = type("D", (), {})()
    data.events = [Event() for _ in validity]
    data.invalid_events = np.array(validity, dtype=bool)
    data.locations = np.array([[1.0, 2.0, 3.0] for _ in validity], dtype=float)
    data.origin_times = np.array([10.0 for _ in validity], dtype=float)
    data.time_refs = np.array([1000.0 for _ in validity], dtype=float)
    data.fixed_depth_mask = np.array([False for _ in validity], dtype=bool)
    data.stations = [("NW", "ST")]
    data.n_stations = len(data.stations)
    data.arrivals = np.array([[12.0] for _ in validity], dtype=float)
    data.theoretical_travel_times = np.array([[1.0] for _ in validity], dtype=float)
    data.distance_W = np.ones(len(validity) * data.n_stations, dtype=float)
    return data


def test_exporter_outputs_catalog():
    config = type("C", (), {"phases": ["P"], "n_phases": 1})()
    data = build_test_event_data([False])
    error_results = ErrorResults(
        spatial_covariances=np.zeros((1, 2, 2)),
        principal_std=np.array([[1.0, 0.8, 0.5]], dtype=float),
        principal_axes=np.eye(3)[None, :, :],
        erz_erh=np.array([[0.02, 0.01]], dtype=float),
    )

    exporter = OriginExporter(config=config, data=data)

    catalog = exporter.finalize_origins(error_results, as_catalog=True)
    assert isinstance(catalog, Catalog)
    assert len(catalog.events) == 1
    assert len(catalog.events[0].origins) == 1

    origin = catalog.events[0].origins[0]
    assert origin.latitude == 1.0
    assert origin.longitude == 2.0
    assert origin.depth == 3000.0
    assert origin.time == 1010.0
    assert len(origin.arrivals) == 1
    assert origin.arrivals[0].phase == "P"
    assert origin.arrivals[0].time_residual == 1.0


def test_exporter_outputs_events_list():
    config = type("C", (), {"phases": ["P"], "n_phases": 1})()
    error_results = ErrorResults(
        spatial_covariances=np.zeros((1, 2, 2)),
        principal_std=np.array([[1.0, 0.8, 0.5]], dtype=float),
        principal_axes=np.eye(3)[None, :, :],
        erz_erh=np.array([[0.02, 0.01]], dtype=float),
    )
    data2 = build_test_event_data([False])
    exporter2 = OriginExporter(config=config, data=data2)
    event_list = exporter2.finalize_origins(error_results, as_catalog=False)
    assert isinstance(event_list, list)
    assert len(event_list) == 1
    assert len(event_list[0].origins) == 1

    origin = event_list[0].origins[0]
    assert origin.latitude == 1.0
    assert origin.longitude == 2.0
    assert origin.depth == 3000.0
    assert origin.time == 1010.0
    assert len(origin.arrivals) == 1
    assert origin.arrivals[0].phase == "P"
    assert origin.arrivals[0].time_residual == 1.0


def test_origin_exporter_event_validation():
    config = type("C", (), {"phases": ["P"], "n_phases": 1})()
    data = build_test_event_data([True, False])
    error_results = ErrorResults(
        spatial_covariances=np.zeros((2, 2, 2)),
        principal_std=np.array([[1.0, 0.8, 0.5], [1.0, 0.8, 0.5]], dtype=float),
        principal_axes=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        erz_erh=np.array([[0.02, 0.01], [0.02, 0.01]], dtype=float),
    )

    exporter = OriginExporter(config=config, data=data)
    events = exporter.finalize_origins(error_results, as_catalog=False)

    assert len(events) == 2
    assert len(events[0].origins) == 0
    assert len(events[1].origins) == 1

    invalid_indices = OriginExporter.report_invalid_events(data)
    assert np.array_equal(invalid_indices, np.array([0]))
