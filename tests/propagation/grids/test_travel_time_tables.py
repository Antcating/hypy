import os
from pathlib import Path

import obspy
import pytest

from hypy.propagation.grid.tau import TauTimeTables


@pytest.fixture(scope="session", params=["taup", "pyrocko"])
def travel_time_tables(request):
    obspy_path = Path(obspy.__file__).parents[0].as_posix()
    model_path = os.path.join(obspy_path, "taup/data/iasp91.tvel")
    model = "iasp91"
    max_depth = 100.0
    max_distance = 1000.0
    depth_step = 10.0
    distance_step = 50.0
    engine = request.param

    def _create_tables(phases):
        return TauTimeTables(
            model_tvel=model_path,
            model_name=model,
            phases=phases,
            max_depth=max_depth,
            max_distance=max_distance,
            depth_step=depth_step,
            distance_step=distance_step,
            engine=engine,
            overwrite=True,
        )

    return _create_tables


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
def test_table_creation(travel_time_tables, phases):
    tables = travel_time_tables(phases)
    assert tables.depth_grid is not None
    assert tables.distance_grid is not None
    for phase in phases:
        assert phase in tables.travel_time_tables


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
def test_compute_travel_time(travel_time_tables, phases):
    tables = travel_time_tables(phases)
    depth = 50.0
    distance = 500.0
    if "P" in phases:
        travel_time_p = tables.compute_travel_time("P", depth, distance)
        assert isinstance(travel_time_p, float)
        assert pytest.approx(travel_time_p, rel=1e-3) == 65.54952023370768
    if "S" in phases:
        travel_time_s = tables.compute_travel_time("S", depth, distance)
        assert isinstance(travel_time_s, float)
        assert pytest.approx(travel_time_s, rel=1e-3) == 117.19924749280987


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
def test_compute_takeoff_angle(travel_time_tables, phases):
    tables = travel_time_tables(phases)
    depth = 50.0
    distance = 500.0
    if "P" in phases:
        takeoff_angle_p = tables.compute_takeoff_angle("P", depth, distance)
        assert isinstance(takeoff_angle_p, float)
        assert pytest.approx(takeoff_angle_p, rel=1e-1) == 89.65376553407343
    if "S" in phases:
        takeoff_angle_s = tables.compute_takeoff_angle("S", depth, distance)
        assert isinstance(takeoff_angle_s, float)
        assert pytest.approx(takeoff_angle_s, rel=1e-1) == 88.82780282349177


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
def test_get_arrival(travel_time_tables, phases):
    tables = travel_time_tables(phases)
    depth = 50.0
    distance = 500.0
    if "P" in phases:
        time_p, angle_p = tables.get_arrival(depth, distance, "P")
        assert isinstance(time_p, float)
        assert pytest.approx(time_p, rel=1e-3) == 65.54952023370768
        assert isinstance(angle_p, float)
        assert pytest.approx(angle_p, rel=1e-1) == 89.65376553407343
    if "S" in phases:
        time_s, angle_s = tables.get_arrival(depth, distance, "S")
        assert isinstance(time_s, float)
        assert pytest.approx(time_s, rel=1e-3) == 117.19924749280987
        assert isinstance(angle_s, float)
        assert pytest.approx(angle_s, rel=1e-1) == 88.82780282349177


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
@pytest.mark.parametrize("depth,distance", [(150.0, 500.0), (50.0, 1500.0)])
def test_get_arrival_out_of_bounds(travel_time_tables, phases, depth, distance):
    tables = travel_time_tables(phases)
    if "P" in phases:
        with pytest.raises(ValueError):
            tables.get_arrival(depth, distance, "P")
    if "S" in phases:
        with pytest.raises(ValueError):
            tables.get_arrival(depth, distance, "S")


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
def test_invalid_phase(travel_time_tables, phases):
    tables = travel_time_tables(phases)
    depth = 50.0
    distance = 500.0
    with pytest.raises(ValueError):
        tables.get_arrival(depth, distance, "X")
