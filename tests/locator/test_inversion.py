"""Unit tests for inversion update constraints."""

import numpy as np

from hypy.locator.config import LocatorConfig, RunConfig
from hypy.locator.data import DataHandler
from hypy.locator.derivatives import DerivativesHandler
from hypy.locator.errors import build_observation_covariance_from_stations
from hypy.locator.inversion import InversionEngine


class _StubTracer:
    name = "stub"
    feature_flags = frozenset()

    def supports_feature(self, feature: str) -> bool:
        return feature in self.feature_flags


def _engine_with_depths(depths_km, fixed=None) -> InversionEngine:
    n_events = len(depths_km)
    config = LocatorConfig(
        phases=("p",),
        min_picks_per_event=1,
        time_error_floor_s=0.1,
        derivative_method="numerical",
        fix_depth_km=None,
    )
    data = DataHandler(config=config)
    data.n_events = n_events
    data.invalid_events = np.zeros(n_events, dtype=bool)
    data.locations = np.column_stack(
        (
            np.full(n_events, 10.0),
            np.full(n_events, 20.0),
            np.asarray(depths_km, dtype=float),
        )
    )
    data.origin_times = np.zeros(n_events, dtype=float)
    data.fixed_depth_mask = list(fixed) if fixed is not None else [False] * n_events
    engine = InversionEngine(
        config=config,
        data=data,
        derivative_engine=DerivativesHandler(
            config=config, data=data, propagation_engine=_StubTracer()
        ),
        run_config=RunConfig(),
    )
    engine.originally_fixed = np.asarray(data.fixed_depth_mask, dtype=bool).copy()
    engine.freeze_depth_km = np.asarray(depths_km, dtype=float).copy()
    engine.max_depth = 50.0
    return engine


def test_clip_updates_zeros_per_event_fixed_depth():
    """Per-event depth constraints zero dz even when the global fix is off."""
    engine = _engine_with_depths([8.0, 8.0], fixed=[True, False])
    dx = np.array(
        [
            [0.1, 0.0, -3.0, 0.2],
            [0.1, 0.0, -3.0, 0.2],
        ],
        dtype=float,
    )
    clipped = engine._clip_updates(0, dx, max_vertical_km=1.0)
    assert clipped[0, 2] == 0.0
    assert clipped[1, 2] == -1.0


def test_freeze_out_of_bounds_at_seed():
    """A step above the free surface freezes depth at the seed, not at z=0."""
    engine = _engine_with_depths([10.0, 10.0])
    dx = np.array(
        [
            [0.0, 0.0, -12.0, 0.4],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    newly = engine._freeze_out_of_bounds_depths(dx)
    assert newly.tolist() == [True, False]
    assert engine.data.fixed_depth_mask == [True, False]
    assert dx[0, 2] == 0.0
    engine.data.locations[0, 2] = 0.0
    frozen = (
        np.asarray(engine.data.fixed_depth_mask, dtype=bool) & ~engine.originally_fixed
    )
    engine.data.locations[frozen, 2] = engine.freeze_depth_km[frozen]
    np.testing.assert_allclose(engine.data.locations[:, 2], [10.0, 10.0])


def test_freeze_out_of_bounds_in_place():
    """unresolved_depth='current' freezes at the hypocenter when it left bounds."""
    engine = _engine_with_depths([4.0, 10.0])
    engine.unresolved_depth = "current"
    engine.freeze_depth_km = np.array([10.0, 10.0], dtype=float)
    dx = np.array(
        [
            [0.0, 0.0, -12.0, 0.4],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    newly = engine._freeze_out_of_bounds_depths(dx)
    assert newly.tolist() == [True, False]
    np.testing.assert_allclose(engine.freeze_depth_km, [4.0, 10.0])
    engine.data.locations[0, 2] = 0.0
    frozen = (
        np.asarray(engine.data.fixed_depth_mask, dtype=bool) & ~engine.originally_fixed
    )
    engine.data.locations[frozen, 2] = engine.freeze_depth_km[frozen]
    np.testing.assert_allclose(engine.data.locations[:, 2], [4.0, 10.0])


def test_freeze_out_of_bounds_too_deep():
    """With unresolved_depth set, a step below max_depth freezes immediately."""
    engine = _engine_with_depths([40.0])
    engine.unresolved_depth = "current"
    dx = np.array([[0.0, 0.0, 15.0, 0.0]], dtype=float)
    newly = engine._freeze_out_of_bounds_depths(dx, iteration_index=0)
    assert newly.tolist() == [True]
    assert engine.data.fixed_depth_mask == [True]
    assert dx[0, 2] == 0.0
    engine.data.locations[0, 2] = 55.0
    frozen = (
        np.asarray(engine.data.fixed_depth_mask, dtype=bool) & ~engine.originally_fixed
    )
    engine.data.locations[frozen, 2] = engine.freeze_depth_km[frozen]
    np.testing.assert_allclose(engine.data.locations[:, 2], [40.0])


def test_depth_elimination_after_window():
    """When unresolved_depth is None, a deep step freezes after the window."""
    engine = _engine_with_depths([40.0])
    engine.unresolved_depth = None
    dx = np.array([[0.0, 0.0, 15.0, 0.0]], dtype=float)
    newly = engine._freeze_out_of_bounds_depths(dx, iteration_index=21)
    assert newly.tolist() == [True]
    assert engine.data.fixed_depth_mask == [True]
    assert dx[0, 2] == 0.0
    engine.data.locations[0, 2] = 55.0
    frozen = (
        np.asarray(engine.data.fixed_depth_mask, dtype=bool) & ~engine.originally_fixed
    )
    engine.data.locations[frozen, 2] = engine.freeze_depth_km[frozen]
    np.testing.assert_allclose(engine.data.locations[:, 2], [40.0])


def test_weighted_rms_uses_step_weights():
    """WRMS is sqrt(sum((w r)^2) / sum(w^2)); a zero weight drops that row."""
    engine = _engine_with_depths([10.0])
    residuals = np.array([1.0, -2.0, 100.0], dtype=float)
    rows = np.array([0, 1, 2])
    weights = np.array([1.0, 0.5, 0.0], dtype=float)
    expected = np.sqrt((1.0 + 1.0) / (1.0 + 0.25))
    assert np.isclose(engine._weighted_rms(residuals, rows, weights), expected)

    residuals[2] = np.nan
    assert np.isclose(engine._weighted_rms(residuals, rows, weights), expected)

    residuals[0] = np.nan
    assert engine._weighted_rms(residuals, rows, weights) == np.inf


def _full_covariance_engine(correlation_length_km: float):
    """One event, four stations on a line; station 2 is residual-weighted out."""
    engine = _engine_with_depths([10.0])
    engine.run_config = RunConfig(
        full_data_covariance=True, correlation_length_km=correlation_length_km
    )
    data = engine.data
    data.n_stations = 4
    data.arrival_time_errors = np.array([[0.1, 0.2, 0.1, 0.3]])
    data.distance_W = np.array([1.0, 0.5, 0.8, 1.0])
    data.residual_W = np.array([1.0, 1.0, 0.0, 0.9])
    x_km = np.array([0.0, 1.0, 2.5, 4.0])
    engine.station_distances_km = np.abs(np.subtract.outer(x_km, x_km))
    row_W = data.distance_W * data.residual_W / data.arrival_time_errors[0]
    return engine, np.arange(4), row_W


def test_full_covariance_whitener_matches_gls_weight():
    """M^T M = D C^-1 D on weighted rows, and WRMS is the matching Mahalanobis norm."""
    engine, rows, row_W = _full_covariance_engine(correlation_length_km=2.0)
    ((pos, event_idx, whitener),) = engine._event_whiteners(rows, row_W)
    assert pos.tolist() == [0, 1, 3] and event_idx == 0

    data = engine.data
    C = build_observation_covariance_from_stations(
        data.arrival_time_errors[0, pos],
        pos,
        np.zeros(3, dtype=int),
        engine.station_distances_km,
        2.0,
    )
    D = np.diag(data.distance_W[pos] * data.residual_W[pos])
    expected = D @ np.linalg.inv(C) @ D
    np.testing.assert_allclose(whitener.T @ whitener, expected, rtol=1e-8)

    residuals = np.array([0.05, -0.1, 99.0, 0.2])
    r = residuals[pos]
    wrms = engine._weighted_rms(residuals, rows, row_W, [(pos, 0, whitener)])
    assert np.isclose(wrms, np.sqrt(r @ expected @ r / np.trace(expected)))


def test_full_covariance_reduces_to_diagonal_without_correlation():
    """Uncorrelated stations give the existing diagonal row weights and WRMS."""
    engine, rows, row_W = _full_covariance_engine(correlation_length_km=1e-6)
    whiteners = engine._event_whiteners(rows, row_W)
    ((pos, _, whitener),) = whiteners
    np.testing.assert_allclose(whitener, np.diag(row_W[pos]), rtol=1e-5, atol=1e-12)

    residuals = np.array([0.05, -0.1, 99.0, 0.2])
    assert np.isclose(
        engine._weighted_rms(residuals, rows, row_W, whiteners),
        engine._weighted_rms(residuals, rows, row_W),
        rtol=1e-5,
    )


def test_depth_elimination_before_window():
    """When unresolved_depth is None, a deep step stays free before the window."""
    engine = _engine_with_depths([40.0])
    engine.unresolved_depth = None
    dx = np.array([[0.0, 0.0, 15.0, 0.0]], dtype=float)
    newly = engine._freeze_out_of_bounds_depths(dx, iteration_index=5)
    assert not np.any(newly)
    assert engine.data.fixed_depth_mask == [False]
    assert dx[0, 2] == 15.0
