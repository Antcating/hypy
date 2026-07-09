import pytest

from hypy.locator import config


@pytest.mark.parametrize(
    "distance_start_km,distance_end_km,weight_start,weight_end,expected",
    [
        # No distance weighting when all parameters are None
        (None, None, None, None, (False, None, None)),
        # No distance weighting when some parameters are None
        (0.0, None, 0.5, 0.5, (False, None, None)),
        (0.0, 100.0, None, 0.5, (False, None, None)),
        (0.0, 100.0, 0.5, None, (False, None, None)),
        # Valid case
        (50.0, 100.0, 1 / 11, 1 / 101, (True, 0.02, 0)),
    ],
)
def test_configure_distance_weighting_valid(
    distance_start_km, distance_end_km, weight_start, weight_end, expected
):
    result = config.configure_distance_weighting(
        distance_start_km, distance_end_km, weight_start, weight_end
    )
    print(result)
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
    ],
)
def test_configure_distance_weighting_invalid(
    distance_start_km, distance_end_km, weight_start, weight_end
):
    with pytest.raises(ValueError):
        config.configure_distance_weighting(
            distance_start_km, distance_end_km, weight_start, weight_end
        )
