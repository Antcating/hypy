import pytest

from hypy.locator import config


def build_run_config_args() -> dict:
    return {
        "distance_weighting_enabled": False,
        "k": None,
        "x0": None,
        "full_data_covariance": False,
        "correlation_length_km": 1.0,
        "error_coefficient": 1.0,
    }


def test_validate_full_data_covariance():
    """Test that the full data covariance is correctly validated."""
    args = build_run_config_args()
    args["full_data_covariance"] = True
    cfg = config.RunConfig(**args)
    assert cfg.full_data_covariance


@pytest.mark.parametrize("correlation_length_km", [0.0, -1.0, None, False, "INVALID"])
def test_validate_correlation_length_km_invalid(correlation_length_km):
    """Test that the correlation length is correctly validated."""
    print(f"correlation_length_km: {correlation_length_km}")
    args = build_run_config_args()
    args["full_data_covariance"] = True
    args["correlation_length_km"] = correlation_length_km
    with pytest.raises(ValueError):
        config.RunConfig(**args)


@pytest.mark.parametrize("error_coefficient", [-1.0, None, "INVALID"])
def test_validate_error_coefficient_invalid(error_coefficient):
    """Test that the error coefficient is correctly validated."""
    args = build_run_config_args()
    args["full_data_covariance"] = True
    args["error_coefficient"] = error_coefficient
    with pytest.raises(ValueError):
        config.RunConfig(**args)


@pytest.mark.parametrize(
    "solver", ["lsmr", "least_squares", "INVALID", 1, None, True, False]
)
def test_validate_solver_invalid(solver):
    """Test that the solver is correctly validated."""
    args = build_run_config_args()
    args["solver"] = solver
    with pytest.raises(ValueError):
        config.RunConfig(**args)


@pytest.mark.parametrize("solver", ["lsqr", "svd"])
def test_validate_solver_valid(solver):
    """Test that the solver is correctly validated."""
    args = build_run_config_args()
    args["solver"] = solver
    cfg = config.RunConfig(**args)
    assert cfg.solver == solver.lower()
