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


@pytest.mark.parametrize("num_trials", [0, -1, 1.5, None, True, "INVALID"])
def test_validate_num_trials_invalid(num_trials):
    """Heuristic trial budget must be an integer >= 1."""
    args = build_run_config_args()
    args["num_trials"] = num_trials
    with pytest.raises(ValueError):
        config.RunConfig(**args)


def test_validate_num_trials_valid():
    """A positive integer trial budget is stored as ``int``."""
    args = build_run_config_args()
    args["num_trials"] = 1
    cfg = config.RunConfig(**args)
    assert cfg.num_trials == 1


@pytest.mark.parametrize("unresolved_depth", ["origin", "INVALID", 1, True, False])
def test_validate_unresolved_depth_invalid(unresolved_depth):
    """Test that unresolved_depth is correctly validated."""
    args = build_run_config_args()
    args["unresolved_depth"] = unresolved_depth
    with pytest.raises(ValueError):
        config.RunConfig(**args)


@pytest.mark.parametrize("unresolved_depth", ["current", "seed", "CURRENT", "Seed"])
def test_validate_unresolved_depth_valid(unresolved_depth):
    """Test that unresolved_depth is stored lowercase."""
    args = build_run_config_args()
    args["unresolved_depth"] = unresolved_depth
    cfg = config.RunConfig(**args)
    assert cfg.unresolved_depth == unresolved_depth.lower()


def test_validate_unresolved_depth_none():
    """None disables seed/current freeze and uses depth elimination."""
    args = build_run_config_args()
    args["unresolved_depth"] = None
    cfg = config.RunConfig(**args)
    assert cfg.unresolved_depth is None
