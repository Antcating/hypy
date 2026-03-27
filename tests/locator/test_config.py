import pytest

from hypy.locator import config


@pytest.mark.parametrize(
    "mode,engine,expected",
    [
        ("ray", "taup", ("ray", "taup")),
        ("table", "pyrocko", ("table", "pyrocko")),
        ("table", "fmm", ("table", "fmm")),
    ],
)
def test_validate_operating_regime_valid(mode, engine, expected):
    result = config.validate_operating_regime(mode, engine)
    assert result == expected


@pytest.mark.parametrize(
    "mode,engine",
    [
        ("invalid_mode", "taup"),
        ("ray", "invalid_engine"),
        ("table", "invalid_engine"),
    ],
)
def test_validate_operating_regime_invalid(mode, engine):
    with pytest.raises(ValueError):
        config.validate_operating_regime(mode, engine)


@pytest.mark.parametrize(
    "solver,expected",
    [
        ("lsqr", "lsqr"),
        ("svd", "svd"),
        ("LSQR", "lsqr"),  # case-insensitive
    ],
)
def test_validate_solver_valid(solver, expected):
    result = config.validate_solver(solver)
    assert result == expected


@pytest.mark.parametrize(
    "solver",
    [
        "invalid_solver",
        "",
    ],
)
def test_validate_solver_invalid(solver):
    with pytest.raises(ValueError):
        config.validate_solver(solver)


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


@pytest.mark.parametrize(
    "max_depth,depth_step",
    [
        (100.0, 10.0),
        (50.0, 5.0),
    ],
)
def test_validate_precompute_args_valid(max_depth, depth_step):
    # Should not raise an exception
    config.validate_precompute_args(max_depth, depth_step)


@pytest.mark.parametrize(
    "max_depth,depth_step",
    [
        (None, None),
        (100.0, None),
        (None, 10.0),
    ],
)
def test_validate_precompute_args_invalid(max_depth, depth_step):
    with pytest.raises(ValueError):
        config.validate_precompute_args(max_depth, depth_step)
