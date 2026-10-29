import numpy as np
import pytest
from obspy.core.event import Catalog, Event

from hypy.locator.export import finalize_origins
from hypy.locator.types import ErrorResults


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
    data.station_time_corrections = np.zeros(data.n_stations, dtype=float)
    data.distance_W = np.ones(len(validity) * data.n_stations, dtype=float)
    data.residual_W = np.ones(len(validity) * data.n_stations, dtype=float)
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

    catalog = finalize_origins(config, data, error_results)
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
    assert origin.arrivals[0].time_correction == 0.0


def test_exporter_subtracts_station_time_correction():
    """Exported residual and time_correction include the station delay."""
    config = type("C", (), {"phases": ["P"], "n_phases": 1})()
    data = build_test_event_data([False])
    data.station_time_corrections = np.array([0.2], dtype=float)
    error_results = ErrorResults(
        spatial_covariances=np.zeros((1, 2, 2)),
        principal_std=np.array([[1.0, 0.8, 0.5]], dtype=float),
        principal_axes=np.eye(3)[None, :, :],
        erz_erh=np.array([[0.02, 0.01]], dtype=float),
    )

    origin = finalize_origins(config, data, error_results)[0].origins[0]
    assert origin.arrivals[0].time_residual == pytest.approx(0.8)
    assert origin.arrivals[0].time_correction == pytest.approx(0.2)


def test_origin_exporter_event_validation():
    config = type("C", (), {"phases": ["P"], "n_phases": 1})()
    data = build_test_event_data([True, False])
    error_results = ErrorResults(
        spatial_covariances=np.zeros((2, 2, 2)),
        principal_std=np.array([[1.0, 0.8, 0.5], [1.0, 0.8, 0.5]], dtype=float),
        principal_axes=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        erz_erh=np.array([[0.02, 0.01], [0.02, 0.01]], dtype=float),
    )

    events = finalize_origins(config, data, error_results)

    assert len(events) == 2
    assert len(events[0].origins) == 0
    assert len(events[1].origins) == 1

    invalid_indices = np.flatnonzero(data.invalid_events)
    assert np.array_equal(invalid_indices, np.array([0]))


def test_confidence_ellipsoid_quakeml_tait_bryan_roundtrip():
    """Exported ConfidenceEllipsoid reconstructs the 95% NED shape via Rz(az)Ry(pl)Rx(rot)."""
    from scipy.stats import chi2

    from hypy.locator.errors import ellipsoid_axes

    rng = np.random.default_rng(0)
    config = type("C", (), {"phases": ["P"], "n_phases": 1})()
    scale95_3d = float(np.sqrt(chi2.ppf(0.95, df=3)))

    for _ in range(5):
        A = rng.normal(size=(3, 3))
        C_ned = A @ A.T + 0.1 * np.eye(3)
        std_axes, evecs, erz, erh = ellipsoid_axes(C_ned)

        data = build_test_event_data([False])
        error_results = ErrorResults(
            spatial_covariances=C_ned[None, :, :],
            principal_std=std_axes[None, :],
            principal_axes=evecs.T[None, :, :],
            erz_erh=np.array([[erz, erh]], dtype=float),
        )
        ce = (
            finalize_origins(config, data, error_results)[0]
            .origins[0]
            .origin_uncertainty.confidence_ellipsoid
        )

        az = np.radians(ce.major_axis_azimuth)
        pl = np.radians(ce.major_axis_plunge)
        rot = np.radians(ce.major_axis_rotation)
        Rz = np.array(
            [
                [np.cos(az), -np.sin(az), 0.0],
                [np.sin(az), np.cos(az), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        Ry = np.array(
            [
                [np.cos(pl), 0.0, np.sin(pl)],
                [0.0, 1.0, 0.0],
                [-np.sin(pl), 0.0, np.cos(pl)],
            ]
        )
        Rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(rot), -np.sin(rot)],
                [0.0, np.sin(rot), np.cos(rot)],
            ]
        )
        R = Rz @ Ry @ Rx
        d = np.diag(
            [
                ce.semi_major_axis_length**2,
                ce.semi_minor_axis_length**2,
                ce.semi_intermediate_axis_length**2,
            ]
        )
        S_reconstructed = R @ d @ R.T
        S_expected = C_ned * (scale95_3d * 1000.0) ** 2
        assert np.allclose(S_reconstructed, S_expected, rtol=1e-10, atol=1e-6)
