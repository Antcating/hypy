import pytest

from hypy.locator import config


def build_config_args() -> dict:
    return {
        "phases": ("P", "S"),
        "min_picks_per_event": 4,
        "default_time_error": 0.1,
        "derivative_method": "numerical",
        "fix_depth_km": None,
        "precalculate_velocities": False,
        "derivative_delta_km": 0.5,
        "dim": 4,
    }


@pytest.mark.parametrize("phases", [("P", "S"), ("P", "S", "PsP")])
def test_n_phases(phases):
    """Test that the number of phases is correctly set."""
    args = build_config_args()
    args["phases"] = phases
    cfg = config.LocatorConfig(**args)
    assert cfg.n_phases == len(phases)


@pytest.mark.parametrize("derivative_method", ["numerical", "analytical"])
def test_validate_derivative_method_valid(derivative_method):
    """Test that the derivative method is correctly validated."""
    args = build_config_args()
    args["derivative_method"] = derivative_method
    cfg = config.LocatorConfig(**args)
    assert cfg.derivative_method == derivative_method.lower()


@pytest.mark.parametrize("derivative_method", [0, None, 1.0, True, False, "INVALID"])
def test_validate_derivative_method_invalid(derivative_method):
    """Test that the derivative method is correctly validated."""
    args = build_config_args()
    args["derivative_method"] = derivative_method
    with pytest.raises(ValueError):
        config.LocatorConfig(**args)
