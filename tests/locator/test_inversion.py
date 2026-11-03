"""Unit tests for inversion update constraints."""

import numpy as np

from hypy.locator.config import LocatorConfig, RunConfig
from hypy.locator.data import DataHandler
from hypy.locator.derivatives import DerivativesHandler
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


def test_solve_matches_damped_least_squares():
    """Step matches (JᵀWJ + λ²I)δ = JᵀWr, and a zero-weight NaN cannot poison it."""
    engine = _engine_with_depths([10.0, 8.0], fixed=[False, True])
    engine.data.n_stations = 6
    engine.damping = 0.4
    rng = np.random.default_rng(0)
    n_obs = engine.data.n_stations * engine.config.n_phases
    jacobian = rng.normal(size=(engine.data.n_events * n_obs, 4))
    residuals = rng.normal(size=jacobian.shape[0])
    residuals[1] = np.nan
    weights = np.concatenate((np.array([1.0, 0.0, 0.5, 1.5, 0.8, 0.3]), np.ones(n_obs)))
    engine.data.jacobian_matrix = jacobian
    dx = engine._solve(np.arange(weights.size), residuals, weights)

    def normal(event, free, w):
        sl = slice(event * n_obs, (event + 1) * n_obs)
        on = w != 0.0
        design = jacobian[sl][on][:, free] * w[on, None]
        rhs = residuals[sl][on] * w[on]
        gram = design.T @ design + engine.damping**2 * np.eye(design.shape[1])
        return np.linalg.solve(gram, design.T @ rhs)

    free = np.array([True, True, True, True])
    np.testing.assert_allclose(dx[0], normal(0, free, weights[:n_obs]), atol=1e-8)
    free[2] = False
    np.testing.assert_allclose(dx[1, free], normal(1, free, weights[n_obs:]), atol=1e-8)
    assert dx[1, 2] == 0.0


def test_iterations_are_passes_over_events():
    """One pass steps every live event. The cap is not restarted per event."""
    engine = _engine_with_depths([10.0, 12.0])
    n_obs = 2
    engine.data.n_stations = 1
    engine.data.residual_W = np.ones(n_obs)
    engine.data.distance_W = np.ones(n_obs)
    engine.data.time_error_W = np.ones(n_obs)
    engine.data.arrivals = np.zeros((2, 1))
    engine.data.theoretical_travel_times = np.zeros((2, 1))
    engine.data.station_time_corrections = np.zeros(1)
    engine.derivative_engine.propagation_engine.max_depth_km = 50.0
    seen: list[tuple[int, int]] = []

    def step(event_idx, step_i, fix_z, conv_threshold, prev_wrms):
        seen.append((event_idx, step_i))
        return False, 1.0

    engine._step_event = step
    engine._derivatives = lambda event_idx: np.array([event_idx], dtype=int)
    engine.run_iterations(max_iterations=2)
    assert seen == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_depth_elimination_before_window():
    """When unresolved_depth is None, a deep step stays free before the window."""
    engine = _engine_with_depths([40.0])
    engine.unresolved_depth = None
    dx = np.array([[0.0, 0.0, 15.0, 0.0]], dtype=float)
    newly = engine._freeze_out_of_bounds_depths(dx, iteration_index=5)
    assert not np.any(newly)
    assert engine.data.fixed_depth_mask == [False]
    assert dx[0, 2] == 15.0
