"""Tests for locator error covariance and uncertainty trends."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from hypy.locator.config import LocatorConfig
from hypy.locator.data import DataHandler
from hypy.locator.errors import ErrorAnalyzer
from hypy.locator.types import InversionResults


def _build_diagonal_design(counts: list[int]) -> np.ndarray:
    """Build a design matrix whose normal matrix is diagonal with known entries."""
    rows = []
    dim = len(counts)
    for col, n_repeat in enumerate(counts):
        basis = np.zeros(dim, dtype=float)
        basis[col] = 1.0
        rows.extend([basis.copy() for _ in range(n_repeat)])
    return np.vstack(rows)


def _build_analyzer_for_single_event(
    n_stations: int, n_phases: int = 1
) -> ErrorAnalyzer:
    config = cast(LocatorConfig, SimpleNamespace(dim=4, n_phases=n_phases))
    data = SimpleNamespace(
        n_events=1,
        n_stations=n_stations,
        invalid_events=np.array([False], dtype=bool),
        arrival_time_errors=np.full((1, n_stations * n_phases), 0.1, dtype=float),
    )
    return ErrorAnalyzer(config=config, data=cast(DataHandler, data))


def test_compute_event_errors_matches_theoretical_closed_form() -> None:
    """Match event covariance against a closed-form diagonal solution."""
    # Choose per-parameter row counts so A^T A is diagonal with unique values.
    counts = [9, 4, 1, 25]
    dim = 4
    damping = 0.5

    A_e = _build_diagonal_design(counts)
    r_e = np.full(A_e.shape[0], 0.3, dtype=float)
    pick_errs_e = np.full(A_e.shape[0], 0.4, dtype=float)

    err = ErrorAnalyzer._compute_event_errors(
        design_matrix=A_e,
        residuals=r_e,
        pick_errors=pick_errs_e,
        damping=damping,
        dim=dim,
    )

    assert err is not None

    sigma2 = 0.3**2 + 0.4**2
    expected_diag = sigma2 / (np.asarray(counts, dtype=float) + damping**2)
    expected_c = np.diag(expected_diag)

    assert np.allclose(err["C_full"], expected_c, atol=1e-12)
    assert np.allclose(err["C_xyz"], expected_c[:3, :3], atol=1e-12)

    # Principal std are sorted descending by variance from Cxyz.
    expected_std_sorted = np.sqrt(np.sort(expected_diag[:3])[::-1])
    assert np.allclose(err["principal_std_km_1sigma"], expected_std_sorted, atol=1e-12)

    # With diagonal Cxyz, ERZ and ERH reduce to z-std and max(x,y)-std.
    expected_erz = float(np.sqrt(expected_diag[2]))
    expected_erh = float(np.sqrt(max(expected_diag[0], expected_diag[1])))
    assert np.isclose(err["ERZ_km_1sigma"], expected_erz, atol=1e-12)
    assert np.isclose(err["ERH_km_1sigma"], expected_erh, atol=1e-12)
    assert err["n_obs_used"] == int(np.sum(counts))


def test_ellipsoid_axes_returns_expected_principal_std_and_scale() -> None:
    """Check error ellipse principal axes and radius for anisotropic covariance."""
    Cxyz = np.array([[5.0, 3.0], [3.0, 5.0]], dtype=float)

    std_axes, erz, erh = ErrorAnalyzer._ellipsoid_axes(Cxyz)

    # Eigenvalues are [8, 2], so 1-sigma std are [sqrt(8), sqrt(2)] sorted descending.
    expected_std = np.array([np.sqrt(8.0), np.sqrt(2.0)], dtype=float)
    assert np.allclose(std_axes, expected_std, atol=1e-12)

    assert np.isnan(erz)
    assert np.isclose(erh, expected_std[0], atol=1e-12)


def test_ellipsoids_1sigma_variance_checks() -> None:
    """Check that output uncertainties match closed-form expected values."""
    config = cast(LocatorConfig, SimpleNamespace(dim=4, n_phases=1))
    data = SimpleNamespace(
        n_events=1,
        n_stations=4,
        invalid_events=np.array([False], dtype=bool),
        arrival_time_errors=np.full((1, 4), 0.1, dtype=float),
    )

    analyzer = ErrorAnalyzer(config=config, data=cast(DataHandler, data))

    # design matrix *without* cross correlation: 4 obs for 4 parameters.
    A_final = np.eye(4, dtype=float)
    residuals_flat = np.full(4, 0.2, dtype=float)

    inv = InversionResults(
        design_matrix=A_final,
        valid_rows=np.arange(4, dtype=int),
        residuals_flat=residuals_flat,
        damping=0.5,
    )

    out = analyzer.compute_error_ellipsoids(inv)

    # Analytical values from C = sigma2 * inv(A^T A + lambda^2 I)
    sigma2 = 0.2**2 + 0.1**2
    expected_var = sigma2 / (1.0 + 0.5**2)

    expected_cov_full = np.diag(
        [expected_var, expected_var, expected_var, expected_var]
    )
    expected_cov_xyz = np.diag([expected_var, expected_var, expected_var])

    assert np.allclose(out.covariances[0], expected_cov_full, atol=1e-12)
    assert np.allclose(out.spatial_covariances[0], expected_cov_xyz, atol=1e-12)

    expected_std = np.sqrt(expected_var)
    assert np.allclose(
        out.principal_std[0],
        np.array([expected_std, expected_std, expected_std]),
        atol=1e-12,
    )
    assert np.isclose(out.erz_erh[0, 0], expected_std, atol=1e-12)
    assert np.isclose(out.erz_erh[0, 1], expected_std, atol=1e-12)


def test_error_decreases_with_more_stations() -> None:
    """Show uncertainty decreases as station count increases."""
    damping = 0.2
    residual_level = 0.3
    pick_error = 0.1

    mean_std = []
    expected_mean_std = []

    # Use multiples of 4 so each parameter gets the same number of constraints.
    for n_stations in (4, 8, 12):
        analyzer = _build_analyzer_for_single_event(n_stations=n_stations)

        counts = [n_stations // 4] * 4
        design_matrix = _build_diagonal_design(counts)
        residuals = np.full(design_matrix.shape[0], residual_level, dtype=float)
        analyzer.data.arrival_time_errors[:] = pick_error

        inv = InversionResults(
            design_matrix=design_matrix,
            valid_rows=np.arange(design_matrix.shape[0], dtype=int),
            residuals_flat=residuals,
            damping=damping,
        )
        out = analyzer.compute_error_ellipsoids(inv)

        mean_std.append(float(np.mean(out.principal_std[0])))

        sigma2 = residual_level**2 + pick_error**2
        per_axis_std = np.sqrt(sigma2 / (counts[0] + damping**2))
        expected_mean_std.append(float(per_axis_std))

    assert mean_std[0] > mean_std[1] > mean_std[2]
    assert np.allclose(mean_std, expected_mean_std, atol=1e-12)


def test_error_decreases_with_smaller_residuals() -> None:
    """Show uncertainty decreases as residual level decreases."""
    damping = 0.2
    n_stations = 8
    pick_error = 0.05

    analyzer = _build_analyzer_for_single_event(n_stations=n_stations)
    counts = [n_stations // 4] * 4
    design_matrix = _build_diagonal_design(counts)
    analyzer.data.arrival_time_errors[:] = pick_error

    mean_std = []
    expected_mean_std = []

    for residual_level in (0.4, 0.2, 0.1):
        residuals = np.full(design_matrix.shape[0], residual_level, dtype=float)
        inv = InversionResults(
            design_matrix=design_matrix,
            valid_rows=np.arange(design_matrix.shape[0], dtype=int),
            residuals_flat=residuals,
            damping=damping,
        )
        out = analyzer.compute_error_ellipsoids(inv)

        mean_std.append(float(np.mean(out.principal_std[0])))

        sigma2 = residual_level**2 + pick_error**2
        per_axis_std = np.sqrt(sigma2 / (counts[0] + damping**2))
        expected_mean_std.append(float(per_axis_std))

    assert mean_std[0] > mean_std[1] > mean_std[2]
    assert np.allclose(mean_std, expected_mean_std, atol=1e-12)


def test_compute_event_errors_returns_none_on_input_shape_guards() -> None:
    """Return None for invalid matrix/vector shapes before algebra starts."""
    assert (
        ErrorAnalyzer._compute_event_errors(
            design_matrix=np.ones((4, 1), dtype=float),
            residuals=np.ones(3, dtype=float),
            pick_errors=np.ones(3, dtype=float),
            damping=0.1,
            dim=1,
        )
        is None
    )
    assert (
        ErrorAnalyzer._compute_event_errors(
            design_matrix=np.zeros((0, 4), dtype=float),
            residuals=np.zeros(0, dtype=float),
            pick_errors=np.zeros(0, dtype=float),
            damping=0.1,
            dim=4,
        )
        is None
    )
    out = ErrorAnalyzer._compute_event_errors(
        design_matrix=np.eye(4, dtype=float),
        residuals=np.ones(4, dtype=float),
        pick_errors=np.array([np.nan], dtype=float),
        damping=0.1,
        dim=4,
    )
    assert out is not None


def test_compute_event_errors_row_filter_and_sigma_guards() -> None:
    """Cover finite-row filtering and non-finite sigma guard paths."""
    # All rows become invalid -> returns None.
    assert (
        ErrorAnalyzer._compute_event_errors(
            design_matrix=np.full((3, 4), np.nan, dtype=float),
            residuals=np.array([1.0, 2.0, 3.0], dtype=float),
            pick_errors=np.array([0.1, 0.1, 0.1], dtype=float),
            damping=0.1,
            dim=4,
        )
        is None
    )

    # Mismatched pick_errs_e length goes through else branch.
    out = ErrorAnalyzer._compute_event_errors(
        design_matrix=np.eye(4, dtype=float),
        residuals=np.ones(4, dtype=float),
        pick_errors=np.array([np.nan], dtype=float),
        damping=0.1,
        dim=4,
    )
    assert out is not None


def test_compute_event_errors_handles_linalg_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise inverse pseudo-inverse fallback and eigh failure handling."""
    A_e = np.eye(4, dtype=float)
    r_e = np.full(4, 0.2, dtype=float)
    pick_errs_e = np.full(4, 0.1, dtype=float)

    original_inv = np.linalg.inv

    def raising_inv(x: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(np.linalg, "inv", raising_inv)
    out = ErrorAnalyzer._compute_event_errors(A_e, r_e, pick_errs_e, damping=0.3, dim=4)
    assert out is not None
    monkeypatch.setattr(np.linalg, "inv", original_inv)

    def raising_eigh(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(np.linalg, "eigh", raising_eigh)
    assert (
        ErrorAnalyzer._compute_event_errors(A_e, r_e, pick_errs_e, damping=0.3, dim=4)
        is None
    )


def test_compute_event_errors_additional_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover sigma2 non-finite, spatial_dim<=0, and non-finite Cxyz branches."""
    A_e = np.eye(4, dtype=float)
    r_e = np.full(4, 0.2, dtype=float)
    pick_errs_e = np.full(4, 0.1, dtype=float)

    original_nanmedian = np.nanmedian
    monkeypatch.setattr(np, "nanmedian", lambda x: np.nan)
    assert ErrorAnalyzer._compute_event_errors(A_e, r_e, pick_errs_e, 0.1, 4) is None

    assert (
        ErrorAnalyzer._compute_event_errors(
            design_matrix=np.zeros((1, 0), dtype=float),
            residuals=np.zeros(1, dtype=float),
            pick_errors=np.zeros(1, dtype=float),
            damping=0.1,
            dim=0,
        )
        is None
    )

    monkeypatch.setattr(np, "nanmedian", original_nanmedian)
    monkeypatch.setattr(
        np.linalg,
        "inv",
        lambda x: np.full((4, 4), np.nan, dtype=float),
    )
    assert ErrorAnalyzer._compute_event_errors(A_e, r_e, pick_errs_e, 0.1, 4) is None


def test_compute_event_errors_spatial_dim_2_and_1_branches() -> None:
    """Cover ERZ/ERH behavior for 2D and 1D spatial blocks."""
    out2 = ErrorAnalyzer._compute_event_errors(
        design_matrix=np.eye(3, dtype=float),
        residuals=np.full(3, 0.2, dtype=float),
        pick_errors=np.full(3, 0.1, dtype=float),
        damping=0.0,
        dim=3,
    )
    assert out2 is not None
    assert np.isnan(out2["ERZ_km_1sigma"])
    assert np.isfinite(out2["ERH_km_1sigma"])

    out1 = ErrorAnalyzer._compute_event_errors(
        design_matrix=np.eye(1, dtype=float),
        residuals=np.full(1, 0.2, dtype=float),
        pick_errors=np.full(1, 0.1, dtype=float),
        damping=0.0,
        dim=1,
    )
    assert out1 is not None
    assert np.isnan(out1["ERZ_km_1sigma"])
    assert np.isfinite(out1["ERH_km_1sigma"])


def test_compute_error_ellipsoids_validation_errors() -> None:
    """Raise ValueError for malformed global inversion array shapes."""
    analyzer = _build_analyzer_for_single_event(n_stations=4)

    with pytest.raises(ValueError, match="A_final must be a 2D array"):
        analyzer.compute_error_ellipsoids(
            InversionResults(
                design_matrix=np.zeros(4, dtype=float),
                valid_rows=np.arange(4, dtype=int),
                residuals_flat=np.zeros(4, dtype=float),
                damping=0.1,
            )
        )

    with pytest.raises(ValueError, match="columns but expected"):
        analyzer.compute_error_ellipsoids(
            InversionResults(
                design_matrix=np.zeros((4, 5), dtype=float),
                valid_rows=np.arange(4, dtype=int),
                residuals_flat=np.zeros(4, dtype=float),
                damping=0.1,
            )
        )

    with pytest.raises(ValueError, match="has length"):
        analyzer.compute_error_ellipsoids(
            InversionResults(
                design_matrix=np.eye(4, dtype=float),
                valid_rows=np.arange(4, dtype=int),
                residuals_flat=np.zeros(3, dtype=float),
                damping=0.1,
            )
        )


def test_compute_error_ellipsoids_skips_invalid_rows_and_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover out-of-range rows, invalid events, empty rows, and errinfo None branch."""
    config = cast(LocatorConfig, SimpleNamespace(dim=4, n_phases=1))
    data = cast(
        DataHandler,
        SimpleNamespace(
            n_events=2,
            n_stations=2,
            invalid_events=np.array([True, False], dtype=bool),
            arrival_time_errors=np.full((2, 2), 0.1, dtype=float),
        ),
    )
    analyzer = ErrorAnalyzer(config=config, data=data)

    # Global rows: 4, cols: 8 (2 events * dim 4)
    design_matrix = np.zeros((4, 8), dtype=float)
    design_matrix[2, 4] = 1.0
    design_matrix[3, 5] = 1.0
    inv = InversionResults(
        design_matrix=design_matrix,
        valid_rows=np.array([-1, 0, 2, 3, 99], dtype=int),
        residuals_flat=np.full(4, 0.2, dtype=float),
        damping=0.1,
    )

    # Force errinfo None only for the non-invalid event branch.
    monkeypatch.setattr(analyzer, "_compute_event_errors", lambda **_: None)
    out = analyzer.compute_error_ellipsoids(inv)

    assert np.allclose(out.covariances, 0.0)
    assert np.isnan(out.erz_erh).all()


def test_compute_error_ellipsoids_row_partition_branches() -> None:
    """Cover event-index and local-row bounds plus empty-row event branch."""
    config = cast(LocatorConfig, SimpleNamespace(dim=4, n_phases=1))
    data = cast(
        DataHandler,
        SimpleNamespace(
            n_events=1,
            n_stations=2,
            invalid_events=np.array([False], dtype=bool),
            # n_err_cols=1 so local_row==1 is out of bounds.
            arrival_time_errors=np.full((1, 1), 0.1, dtype=float),
        ),
    )
    analyzer = ErrorAnalyzer(config=config, data=data)

    design_matrix = np.zeros((4, 4), dtype=float)
    inv = InversionResults(
        design_matrix=design_matrix,
        # row 3 -> ev=1 out-of-range for n_events=1, row 1 -> local_row=1 out-of-range
        valid_rows=np.array([1, 3], dtype=int),
        residuals_flat=np.zeros(4, dtype=float),
        damping=0.1,
    )
    out = analyzer.compute_error_ellipsoids(inv)
    assert np.allclose(out.covariances, 0.0)
