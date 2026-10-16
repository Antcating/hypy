import pytest

from hypy.locator import config


def build_config_args() -> dict:
    return {
        "phases": ("P", "S"),
        "min_picks_per_event": 4,
        "time_error_floor_s": 0.1,
        "derivative_method": "numerical",
        "fix_depth_km": None,
        "derivative_delta_km": 0.5,
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


def test_search_bounds_all_or_none():
    """Partial search bounds are rejected; a full box is accepted."""
    args = build_config_args()
    args["latitude_min_deg"] = 1.0
    with pytest.raises(ValueError, match="all be set or all be None"):
        config.LocatorConfig(**args)

    args.update(
        {
            "latitude_min_deg": 1.0,
            "latitude_max_deg": 2.0,
            "longitude_min_deg": 3.0,
            "longitude_max_deg": 4.0,
        }
    )
    cfg = config.LocatorConfig(**args)
    assert cfg.latitude_min_deg == 1.0
    assert cfg.longitude_max_deg == 4.0


def test_search_bounds_order():
    """Lat/lon mins must be strictly less than maxes."""
    args = build_config_args()
    args.update(
        {
            "latitude_min_deg": 2.0,
            "latitude_max_deg": 1.0,
            "longitude_min_deg": 3.0,
            "longitude_max_deg": 4.0,
        }
    )
    with pytest.raises(ValueError, match="latitude_min_deg"):
        config.LocatorConfig(**args)


def test_search_bounds_property_and_clamp():
    """``search_bounds`` and ``clamp_horizontal`` expose the configured box."""
    args = build_config_args()
    cfg = config.LocatorConfig(**args)
    assert cfg.search_bounds is None
    assert cfg.clamp_horizontal(9.0, 21.0) == (9.0, 21.0)

    args.update(
        {
            "latitude_min_deg": 9.5,
            "latitude_max_deg": 10.5,
            "longitude_min_deg": 19.5,
            "longitude_max_deg": 20.5,
        }
    )
    cfg = config.LocatorConfig(**args)
    assert cfg.search_bounds == (9.5, 10.5, 19.5, 20.5)
    assert cfg.clamp_horizontal(9.0, 21.0) == (9.5, 20.5)
    assert cfg.clamp_horizontal(10.0, 20.0) == (10.0, 20.0)
