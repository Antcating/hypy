import numpy as np
import pytest

from hypy.utils import weighting


@pytest.mark.parametrize(
    "distance_start_km,distance_end_km,weight_start,weight_end,expected",
    [
        # No distance weighting when all parameters are None
        (None, None, None, None, (False, None, None)),
        # Valid case
        (50.0, 100.0, 1 / 11, 1 / 101, (True, 0.02, 0)),
    ],
)
def test_configure_distance_weighting_valid(
    distance_start_km, distance_end_km, weight_start, weight_end, expected
):
    result = weighting.configure_distance_weighting(
        distance_start_km, distance_end_km, weight_start, weight_end
    )
    assert result == expected


@pytest.mark.parametrize(
    "distance_start_km,distance_end_km,weight_start,weight_end",
    [
        # weight_start is bigger than weight_end
        (0.0, 100.0, 1.0, 0.5),
        # weight_end is bigger than 1.0
        (0.0, 100.0, 0.1, 10.0),
        # distance_weight_start is bigger than distance_weight_end
        (100.0, 0.0, 0.5, 0.5),
        # some parameters are None
        (0.0, None, 0.5, 0.5),
        (0.0, 100.0, None, 0.5),
        (0.0, 100.0, 0.5, None),
    ],
)
def test_configure_distance_weighting_invalid(
    distance_start_km, distance_end_km, weight_start, weight_end
):
    with pytest.raises(ValueError):
        weighting.configure_distance_weighting(
            distance_start_km, distance_end_km, weight_start, weight_end
        )


def test_configure_matches_distance_weight_apply():
    """Configured (k, x0) must reproduce endpoint weights under
    distance_weight."""
    d0, d1 = 50.0, 100.0
    w0, w1 = 1 / 11, 1 / 101
    enabled, k, x0 = weighting.configure_distance_weighting(d0, d1, w0, w1)
    assert enabled
    # distance_weight returns sqrt(w); square to recover the logistic value.
    assert np.isclose(weighting.distance_weight(d0, k=k, x0=x0) ** 2, w0)
    assert np.isclose(weighting.distance_weight(d1, k=k, x0=x0) ** 2, w1)
