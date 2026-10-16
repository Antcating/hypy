import numpy as np
import pytest

from hypy.locator import config
from hypy.utils import weighting


def test_residual_weight_factor_endpoints():
    """Full weight below lo, zero at/above hi, 0.5 at the midpoint."""
    # Force temp2 = residual_rms_floor_s by using tiny residuals for the RMS
    # estimate, then evaluate factors on a residual vector with unit weights.
    base = np.ones(3)
    # With |r| all equal to 0, rmswt=0 -> temp2=residual_rms_floor_s=0.1
    # lo=0.1, hi=0.3 for full/zero factors 1 and 3
    tiny = np.array([1e-12, 1e-12, 1e-12])
    factors_tiny = weighting.residual_weight_factor(
        tiny,
        base,
        residual_rms_floor_s=0.1,
        residual_full_weight_factor=1.0,
        residual_zero_weight_factor=3.0,
    )
    assert np.allclose(factors_tiny, 1.0)

    # Keep RMS below the floor by setting base weights so only a near-zero
    # residual dominates the RMS estimate.
    base = np.array([1.0, 1e-12, 1e-12, 1e-12, 1e-12])
    residuals = np.array([0.0, 0.1, 0.2, 0.3, 0.5])
    sqrt_factor = weighting.residual_weight_factor(
        residuals,
        base,
        residual_rms_floor_s=0.1,
        residual_full_weight_factor=1.0,
        residual_zero_weight_factor=3.0,
    )
    factor = sqrt_factor**2
    # temp2=0.1, lo=0.1, hi=0.3
    assert np.isclose(factor[0], 1.0)  # |r|=0
    assert np.isclose(factor[1], 1.0)  # |r|=lo
    assert np.isclose(factor[2], 0.5)  # midpoint of [0.1, 0.3]
    assert np.isclose(factor[3], 0.0)  # |r|=hi
    assert np.isclose(factor[4], 0.0)  # |r|>hi


def test_residual_weight_factor_shape_mismatch():
    with pytest.raises(ValueError):
        weighting.residual_weight_factor(np.ones(2), np.ones(3))


def test_run_config_residual_weighting_valid():
    cfg = config.RunConfig(
        residual_weighting_enabled=True,
        residual_start_iteration=4,
        residual_rms_floor_s=0.1,
        residual_full_weight_factor=1.0,
        residual_zero_weight_factor=3.0,
    )
    assert cfg.residual_weighting_enabled
    assert cfg.residual_start_iteration == 4
    assert cfg.residual_rms_floor_s == 0.1
    assert cfg.residual_full_weight_factor == 1.0
    assert cfg.residual_zero_weight_factor == 3.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"residual_weighting_enabled": True, "residual_start_iteration": None},
        {"residual_weighting_enabled": True, "residual_start_iteration": 0},
        {"residual_weighting_enabled": True, "residual_start_iteration": -1},
        {
            "residual_weighting_enabled": True,
            "residual_start_iteration": 4,
            "residual_rms_floor_s": 0.0,
        },
        {
            "residual_weighting_enabled": True,
            "residual_start_iteration": 4,
            "residual_rms_floor_s": -0.1,
        },
        {
            "residual_weighting_enabled": True,
            "residual_start_iteration": 4,
            "residual_full_weight_factor": -0.1,
        },
        {
            "residual_weighting_enabled": True,
            "residual_start_iteration": 4,
            "residual_full_weight_factor": 3.0,
            "residual_zero_weight_factor": 3.0,
        },
        {
            "residual_weighting_enabled": True,
            "residual_start_iteration": 4,
            "residual_full_weight_factor": 3.0,
            "residual_zero_weight_factor": 1.0,
        },
        {
            "residual_weighting_enabled": False,
            "residual_start_iteration": 4,
        },
    ],
)
def test_run_config_residual_weighting_invalid(kwargs):
    with pytest.raises(ValueError):
        config.RunConfig(**kwargs)
