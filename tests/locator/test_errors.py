"""Tests for locator error covariance and uncertainty calculations."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import hypy.locator.errors as error_module
from hypy.locator.config import LocatorConfig, RunConfig
from hypy.locator.data import DataHandler
from hypy.locator.errors import (
    build_observation_covariance_from_stations,
    compute_error_ellipsoids,
    compute_event_errors,
    covariance_from_design,
    data_variance,
    ellipsoid_axes,
    whiten_design,
)
from hypy.locator.types import InversionResults


def _build_diagonal_design(counts: list[int]) -> np.ndarray:
    rows = []
    dim = len(counts)
    for col, n_repeat in enumerate(counts):
        basis = np.zeros(dim, dtype=float)
        basis[col] = 1.0
        rows.extend([basis.copy() for _ in range(n_repeat)])
    return np.vstack(rows)


def _build_single_event_inputs(
    n_stations: int,
    n_phases: int = 1,
    full_data_covariance: bool = False,
) -> tuple[LocatorConfig, DataHandler, RunConfig]:
    config = cast(
        LocatorConfig,
        SimpleNamespace(
            dim=4,
            n_phases=n_phases,
        ),
    )
    run_config = RunConfig(
        full_data_covariance=full_data_covariance,
        correlation_length_km=1.0,
    )
    data = cast(
        DataHandler,
        SimpleNamespace(
            n_events=1,
            n_stations=n_stations,
            invalid_events=np.array([False], dtype=bool),
            arrival_time_errors=np.full((1, n_stations * n_phases), 0.1, dtype=float),
            stations_locations=np.zeros((n_stations, 3), dtype=float),
            distance_W=np.ones(n_stations * n_phases, dtype=float),
            residual_W=np.ones(n_stations * n_phases, dtype=float),
        ),
    )
    return config, data, run_config


def test_build_observation_covariance_from_stations_returns_valid_covariance() -> None:
    pick_errors = np.array([0.4, 0.6], dtype=float)
    station_indices = np.array([0, 1], dtype=int)
    phase_indices = np.array([0, 0], dtype=int)
    station_distances_km = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

    covariance = build_observation_covariance_from_stations(
        pick_errors=pick_errors,
        station_indices=station_indices,
        phase_indices=phase_indices,
        station_distances_km=station_distances_km,
        corr_length_km=1.0,
        jitter_fraction=1e-6,
    )

    assert covariance is not None
    assert covariance.shape == (2, 2)
    assert np.allclose(covariance, covariance.T)
    assert covariance[0, 0] > covariance[0, 1]
    assert covariance[0, 1] > 0.0


def test_build_observation_covariance_from_stations_returns_none_for_invalid_inputs() -> (
    None
):
    invalid_args = [
        (
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.zeros((0, 0), dtype=float),
            1.0,
        ),
        (
            np.array([0.1], dtype=float),
            np.array([0, 1], dtype=int),
            np.array([0], dtype=int),
            np.zeros((1, 1), dtype=float),
            1.0,
        ),
        (
            np.array([0.1], dtype=float),
            np.array([0], dtype=int),
            np.array([0, 0], dtype=int),
            np.zeros((1, 1), dtype=float),
            1.0,
        ),
        (
            np.array([0.1], dtype=float),
            np.array([0], dtype=int),
            np.array([0], dtype=int),
            np.ones((2, 1), dtype=float),
            1.0,
        ),
        (
            np.array([0.1], dtype=float),
            np.array([1], dtype=int),
            np.array([0], dtype=int),
            np.eye(1, dtype=float),
            1.0,
        ),
        (
            np.array([0.1], dtype=float),
            np.array([0], dtype=int),
            np.array([0], dtype=int),
            np.eye(1, dtype=float),
            0.0,
        ),
    ]

    for (
        pick_errors,
        station_indices,
        phase_indices,
        station_distances_km,
        corr_length_km,
    ) in invalid_args:
        assert (
            build_observation_covariance_from_stations(
                pick_errors=pick_errors,
                station_indices=station_indices,
                phase_indices=phase_indices,
                station_distances_km=station_distances_km,
                corr_length_km=corr_length_km,
            )
            is None
        )


def test_hypoinverse_data_variance_uses_rderr_and_rms() -> None:
    residuals = np.array([0.1, -0.2, 0.3], dtype=float)

    sigma2, rms = data_variance(
        residuals=residuals,
        model_uncertainty_s=0.2,
        error_coefficient=2.0,
    )

    expected_rms = float(np.sqrt(np.mean(residuals**2)))
    assert np.isclose(rms, expected_rms, atol=1e-12)
    assert np.isclose(sigma2, 0.2**2 + 2.0 * expected_rms**2, atol=1e-12)


def test_data_variance_downweights_residual() -> None:
    residuals = np.array([0.0, 2.0], dtype=float)
    _, wrms = data_variance(residuals, 0.1, weights=np.array([1.0, 0.5]))
    assert np.isclose(wrms, np.sqrt(0.8), atol=1e-12)


def test_data_variance_floor_is_model_uncertainty() -> None:
    """A perfect fit still carries the travel-time model error."""
    sigma2, rms = data_variance(np.zeros(3, dtype=float), 0.15)
    assert rms == 0.0
    assert np.isclose(sigma2, 0.15**2, atol=1e-12)


def test_whiten_design_uses_relative_observation_covariance() -> None:
    design = np.eye(2, dtype=float)
    observation_covariance = np.diag(np.array([0.04, 0.16], dtype=float))

    whitened = whiten_design(
        jacobian_matrix=design,
        observation_covariance=observation_covariance,
        reference_variance=0.04,
    )

    assert whitened is not None
    assert np.allclose(whitened, np.diag([1.0, 0.5]), atol=1e-12)


def test_hypoinverse_covariance_from_design_uses_svd() -> None:
    design = _build_diagonal_design([2, 2])
    covariance = covariance_from_design(
        design_matrix=design,
        sigma2=0.5,
    )

    assert covariance is not None
    assert np.allclose(covariance, np.diag([0.25, 0.25]), atol=1e-12)


def test_ellipsoid_axes_returns_expected_principal_std_and_errors() -> None:
    Cxyz2 = np.array([[5.0, 3.0], [3.0, 5.0]], dtype=float)
    std_axes2, _, erz2, erh2 = ellipsoid_axes(Cxyz2)
    assert np.allclose(
        std_axes2, np.array([np.sqrt(8.0), np.sqrt(2.0)], dtype=float), atol=1e-12
    )
    assert np.isnan(erz2)
    assert np.isclose(erh2, std_axes2[0], atol=1e-12)

    Cxyz3 = np.diag(np.array([4.0, 1.0, 0.25], dtype=float))
    std_axes3, _, erz3, erh3 = ellipsoid_axes(Cxyz3)
    assert np.allclose(std_axes3, np.array([2.0, 1.0, 0.5]), atol=1e-12)
    assert np.isclose(erz3, 0.5, atol=1e-12)
    assert np.isclose(erh3, 2.0, atol=1e-12)

    Cxyz1 = np.array([[0.16]], dtype=float)
    std_axes1, _, erz1, erh1 = ellipsoid_axes(Cxyz1)
    assert np.allclose(std_axes1, np.array([0.4]), atol=1e-12)
    assert np.isnan(erz1)
    assert np.isclose(erh1, 0.4, atol=1e-12)


def test_compute_event_errors_with_diagonal_design_computes_expected_covariance() -> (
    None
):
    A = _build_diagonal_design([2, 2, 2, 2])
    residuals = np.full(A.shape[0], 0.2, dtype=float)
    pick_errors = np.full(A.shape[0], 0.1, dtype=float)

    result = compute_event_errors(
        jacobian_matrix=A,
        residuals=residuals,
        pick_errors=pick_errors,
        dim=4,
        model_uncertainty_s=0.1,
    )

    assert result is not None
    expected_var = (0.1**2 + 0.2**2) / 2.0
    assert np.allclose(result["C_xyz"], np.diag([expected_var] * 3), atol=1e-12)
    assert np.isclose(result["WRMS_s"], 0.2, atol=1e-12)
    assert np.isclose(result["SIGMA2_s2"], 0.1**2 + 0.2**2, atol=1e-12)
    assert result["n_obs_used"] == A.shape[0]


def test_event_error_handling_with_identity_covariance() -> None:
    A = _build_diagonal_design([2, 2, 2, 2])
    residuals = np.full(A.shape[0], 0.2, dtype=float)
    pick_errors = np.full(A.shape[0], 0.1, dtype=float)

    identity_cov = np.eye(A.shape[0], dtype=float)
    result_identity = compute_event_errors(
        jacobian_matrix=A,
        residuals=residuals,
        pick_errors=pick_errors,
        dim=4,
        observation_covariance=identity_cov,
    )
    assert result_identity is not None

    singular_cov = np.ones((A.shape[0], A.shape[0]), dtype=float)
    result_singular = compute_event_errors(
        jacobian_matrix=A,
        residuals=residuals,
        pick_errors=pick_errors,
        dim=4,
        observation_covariance=singular_cov,
    )
    assert result_singular is None


def test_compute_event_errors_with_invalid_shapes() -> None:
    assert (
        compute_event_errors(
            jacobian_matrix=np.ones((4, 1), dtype=float),
            residuals=np.ones(3, dtype=float),
            pick_errors=np.ones(3, dtype=float),
            dim=1,
        )
        is None
    )
    assert (
        compute_event_errors(
            jacobian_matrix=np.zeros((0, 4), dtype=float),
            residuals=np.zeros(0, dtype=float),
            pick_errors=np.zeros(0, dtype=float),
            dim=4,
        )
        is None
    )
    assert (
        compute_event_errors(
            jacobian_matrix=np.zeros((1, 0), dtype=float),
            residuals=np.zeros(1, dtype=float),
            pick_errors=np.zeros(1, dtype=float),
            dim=0,
        )
        is None
    )


def test_compute_event_errors_returns_none_when_no_degrees_of_freedom() -> None:
    A = _build_diagonal_design([1, 1, 1, 1])
    residuals = np.full(A.shape[0], 0.2, dtype=float)
    pick_errors = np.full(A.shape[0], 0.1, dtype=float)

    assert (
        compute_event_errors(
            jacobian_matrix=A,
            residuals=residuals,
            pick_errors=pick_errors,
            dim=5,
        )
        is None
    )


def test_compute_event_errors_supports_exactly_determined_solution() -> None:
    A = _build_diagonal_design([1, 1, 1, 1])
    residuals = np.full(A.shape[0], 0.2, dtype=float)
    pick_errors = np.full(A.shape[0], 0.1, dtype=float)

    result = compute_event_errors(
        jacobian_matrix=A,
        residuals=residuals,
        pick_errors=pick_errors,
        dim=4,
        model_uncertainty_s=0.1,
    )

    assert result is not None
    assert result["n_obs_used"] == 4
    assert np.allclose(result["C_xyz"], np.diag([0.05] * 3), atol=1e-12)


def test_compute_error_ellipsoids_raises_for_malformed_inversion_results() -> None:
    config, data, run_config = _build_single_event_inputs(n_stations=4)

    with pytest.raises(ValueError, match="A_final must be a 2D array"):
        compute_error_ellipsoids(
            config,
            data,
            InversionResults(
                jacobian_matrix=np.zeros(4, dtype=float),
                valid_rows=np.arange(4, dtype=int),
                residuals_flat=np.zeros(4, dtype=float),
            ),
            run_config,
        )

    with pytest.raises(ValueError, match="columns but expected"):
        compute_error_ellipsoids(
            config,
            data,
            InversionResults(
                jacobian_matrix=np.zeros((4, 5), dtype=float),
                valid_rows=np.arange(4, dtype=int),
                residuals_flat=np.zeros(4, dtype=float),
            ),
            run_config,
        )

    with pytest.raises(ValueError, match="has length"):
        compute_error_ellipsoids(
            config,
            data,
            InversionResults(
                jacobian_matrix=np.eye(4, dtype=float),
                valid_rows=np.arange(4, dtype=int),
                residuals_flat=np.zeros(3, dtype=float),
            ),
            run_config,
        )


def test_compute_error_ellipsoids_skips_invalid_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = cast(
        LocatorConfig,
        SimpleNamespace(dim=4, n_phases=1),
    )
    run_config = RunConfig(full_data_covariance=False)
    data = cast(
        DataHandler,
        SimpleNamespace(
            n_events=2,
            n_stations=2,
            invalid_events=np.array([True, False], dtype=bool),
            arrival_time_errors=np.full((2, 2), 0.1, dtype=float),
            stations_locations=np.zeros((2, 3), dtype=float),
            distance_W=np.ones(4, dtype=float),
            residual_W=np.ones(4, dtype=float),
        ),
    )
    jacobian_matrix_weighted = np.zeros((4, 8), dtype=float)
    jacobian_matrix_weighted[2, 4] = 1.0
    jacobian_matrix_weighted[3, 5] = 1.0
    inv = InversionResults(
        jacobian_matrix=np.zeros_like(jacobian_matrix_weighted),
        valid_rows=np.array([0, 2, 3], dtype=int),
        residuals_flat=np.full(4, 0.2, dtype=float),
    )

    monkeypatch.setattr(error_module, "compute_event_errors", lambda **_: None)
    out = compute_error_ellipsoids(config, data, inv, run_config)

    assert np.allclose(out.spatial_covariances, 0.0)
    assert np.isnan(out.erz_erh).all()


def test_compute_error_ellipsoids_uses_full_design_row_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = cast(
        LocatorConfig,
        SimpleNamespace(dim=4, n_phases=1),
    )
    run_config = RunConfig(full_data_covariance=False)
    data = cast(
        DataHandler,
        SimpleNamespace(
            n_events=1,
            n_stations=4,
            invalid_events=np.array([False], dtype=bool),
            arrival_time_errors=np.full((1, 4), 0.1, dtype=float),
            stations_locations=np.zeros((4, 3), dtype=float),
            distance_W=np.ones(4, dtype=float),
            residual_W=np.ones(4, dtype=float),
        ),
    )
    jacobian_matrix_weighted = np.zeros((4, 4), dtype=float)
    jacobian_matrix_weighted[0, 0] = 1.0
    jacobian_matrix_weighted[2, 1] = 1.0
    inv = InversionResults(
        jacobian_matrix=jacobian_matrix_weighted.copy(),
        valid_rows=np.array([0, 2], dtype=int),
        residuals_flat=np.array([0.1, np.nan, 0.2, np.nan], dtype=float),
    )

    captured: dict[str, np.ndarray] = {}

    def fake_compute_event_errors(
        jacobian_matrix: np.ndarray,
        residuals: np.ndarray,
        pick_errors: np.ndarray,
        **kwargs,
    ) -> dict:
        captured["jacobian"] = jacobian_matrix.copy()
        captured["residuals"] = residuals.copy()
        captured["pick_errors"] = pick_errors.copy()
        return {
            "C_xyz": np.zeros((3, 3), dtype=float),
            "principal_std_km_1sigma": np.zeros(3, dtype=float),
            "principal_axes_unit_vectors": np.zeros((3, 3), dtype=float),
            "ERZ_km_1sigma": np.nan,
            "ERH_km_1sigma": np.nan,
            "RMS_s": 0.0,
            "n_obs_used": 2,
        }

    monkeypatch.setattr(error_module, "compute_event_errors", fake_compute_event_errors)
    out = compute_error_ellipsoids(config, data, inv, run_config)

    assert np.allclose(captured["jacobian"], jacobian_matrix_weighted[[0, 2], :])
    assert np.allclose(captured["residuals"], np.array([0.1, 0.2], dtype=float))
    assert np.allclose(captured["pick_errors"], np.array([0.1, 0.1], dtype=float))
    assert np.allclose(out.spatial_covariances, 0.0)


def test_error_ellipsoids_full_covariance_observation() -> None:
    config = cast(
        LocatorConfig,
        SimpleNamespace(
            dim=4,
            n_phases=2,
        ),
    )
    run_config = RunConfig(
        full_data_covariance=True,
        correlation_length_km=10.0,
    )
    data = cast(
        DataHandler,
        SimpleNamespace(
            n_events=1,
            n_stations=3,
            invalid_events=np.array([False], dtype=bool),
            arrival_time_errors=np.array([[0.1, 0.2, 0.1, 0.2, 0.1, 0.2]], dtype=float),
            stations_locations=np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.02, 0.0]],
                dtype=float,
            ),
            distance_W=np.ones(6, dtype=float),
            residual_W=np.ones(6, dtype=float),
        ),
    )
    jacobian_matrix = _build_diagonal_design([2, 2, 1, 1])
    inv = InversionResults(
        jacobian_matrix=jacobian_matrix,
        valid_rows=np.arange(jacobian_matrix.shape[0], dtype=int),
        residuals_flat=np.full(jacobian_matrix.shape[0], 0.1, dtype=float),
    )

    out = compute_error_ellipsoids(config, data, inv, run_config)
    assert out.spatial_covariances.shape == (1, 3, 3)
    assert out.principal_std.shape == (1, 3)
    assert np.isfinite(out.erz_erh[0, 1])
