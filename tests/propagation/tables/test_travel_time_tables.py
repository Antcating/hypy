import os
from pathlib import Path

import obspy
import pytest

from hypy.propagation.tables.travel_time_tables import TravelTimeTables


@pytest.fixture(params=["taup", "pyrocko"])
def travel_time_tables(request):
    obspy_path = Path(obspy.__file__).parents[0].as_posix()
    model_path = os.path.join(obspy_path, "taup/data/iasp91.tvel")
    model = "iasp91"
    phases = ["P", "S"]
    engine = request.param
    max_depth = 100.0
    max_distance = 1000.0
    depth_step = 10.0
    distance_step = 50.0
    tables = TravelTimeTables(
        model_tvel=model_path,
        model=model,
        phases=phases,
        max_depth=max_depth,
        max_distance=max_distance,
        depth_step=depth_step,
        distance_step=distance_step,
        engine=engine,
        overwrite=True,
    )
    return tables


def test_table_creation(travel_time_tables):
    assert travel_time_tables.depth_grid is not None
    assert travel_time_tables.distance_grid is not None
    assert "P" in travel_time_tables.travel_time_tables
    assert "S" in travel_time_tables.travel_time_tables


def test_compute_travel_time(travel_time_tables):
    depth = 50.0
    distance = 500.0
    travel_time_p = travel_time_tables.compute_travel_time("P", depth, distance)
    travel_time_s = travel_time_tables.compute_travel_time("S", depth, distance)
    assert isinstance(travel_time_p, float)
    assert pytest.approx(travel_time_p, rel=1e-3) == 65.54952023370768
    assert isinstance(travel_time_s, float)
    assert pytest.approx(travel_time_s, rel=1e-3) == 117.19924749280987


def test_compute_takeoff_angle(travel_time_tables):
    depth = 50.0
    distance = 500.0
    takeoff_angle_p = travel_time_tables.compute_takeoff_angle("P", depth, distance)
    takeoff_angle_s = travel_time_tables.compute_takeoff_angle("S", depth, distance)
    assert isinstance(takeoff_angle_p, float)
    assert isinstance(takeoff_angle_s, float)
    assert pytest.approx(takeoff_angle_p, rel=1e-1) == 89.65376553407343
    assert pytest.approx(takeoff_angle_s, rel=1e-1) == 88.82780282349177


def test_get_arrival(travel_time_tables):
    depth = 50.0
    distance = 500.0
    time_p, angle_p = travel_time_tables.get_arrival(depth, distance, "P")
    time_s, angle_s = travel_time_tables.get_arrival(depth, distance, "S")
    assert isinstance(time_p, float)
    assert pytest.approx(time_p, rel=1e-3) == 65.54952023370768
    assert isinstance(angle_p, float)
    assert pytest.approx(angle_p, rel=1e-1) == 89.65376553407343
    assert isinstance(time_s, float)
    assert pytest.approx(time_s, rel=1e-3) == 117.19924749280987
    assert isinstance(angle_s, float)
    assert pytest.approx(angle_s, rel=1e-1) == 88.82780282349177


def test_invalid_phase(travel_time_tables):
    depth = 50.0
    distance = 500.0
    with pytest.raises(ValueError):
        travel_time_tables.get_arrival(depth, distance, "X")