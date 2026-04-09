"""Tests for layered travel-time table backends."""

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import obspy
import pytest

from hypy.propagation.layered.grid import TauTravelTimeGrid


@pytest.fixture(scope="session", params=["taup", "pyrocko"])
def travel_time_tables(
    request, overwrite=False
) -> Callable[[list[str], bool], TauTravelTimeGrid]:
    """Build travel-time table factories for the layered grid tests."""
    obspy_path = Path(obspy.__file__).parents[0].as_posix()
    model_path = os.path.join(obspy_path, "taup/data/iasp91.tvel")
    max_depth = 100.0
    max_distance = 1000.0
    depth_step = 10.0
    distance_step = 50.0
    engine = request.param

    def _create_tables(phases, overwrite=False) -> TauTravelTimeGrid:
        return TauTravelTimeGrid(
            model_tvel=model_path,
            phases=phases,
            max_depth=max_depth,
            max_distance=max_distance,
            depth_step=depth_step,
            distance_step=distance_step,
            engine=engine,
            overwrite=overwrite,
        )

    return _create_tables


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
def test_table_creation(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
    phases: list[str],
):
    """Creating tables should populate the expected phase grids."""
    tables = travel_time_tables(phases, True)
    assert tables.depth_grid is not None
    assert tables.distance_grid is not None
    assert tables.travel_time_tables is not None
    for phase in phases:
        assert phase in tables.travel_time_tables


def test_table_reuse(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
):
    """Reusing tables should yield equivalent cached travel-time values."""
    # Create tables with overwrite=True to ensure they are generated
    tables1 = travel_time_tables(["P", "S"], True)
    # Create tables again with overwrite=False to test reuse
    tables2 = travel_time_tables(["P", "S"], False)
    assert tables1.depth_grid is not None
    assert tables1.distance_grid is not None
    assert tables1.travel_time_tables is not None
    assert tables2.depth_grid is not None
    assert tables2.distance_grid is not None
    assert tables2.travel_time_tables is not None
    for phase in ["P", "S"]:
        assert phase in tables1.travel_time_tables
        assert phase in tables2.travel_time_tables
        # Check that the tables are the same object (i.e., reused)
        assert np.array_equal(
            tables1.travel_time_tables[phase], tables2.travel_time_tables[phase]
        )


def test_load_invalid_phases(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
):
    """Unsupported phases should raise a validation error."""
    with pytest.raises(ValueError):
        travel_time_tables(["X"], False)


def test_invalid_engine():
    """Unsupported engines should raise a validation error."""
    with pytest.raises(ValueError):
        obspy_path = Path(obspy.__file__).parents[0].as_posix()
        model_path = os.path.join(obspy_path, "taup/data/iasp91.tvel")
        TauTravelTimeGrid(
            model_tvel=model_path,
            phases=["P"],
            max_depth=100.0,
            max_distance=1000.0,
            depth_step=10.0,
            distance_step=50.0,
            engine="invalid_engine",
            overwrite=True,
        )


def test_save_emply_table(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
    tmp_path: Path,
):
    """Saving an empty table collection should fail."""
    tables: TauTravelTimeGrid = travel_time_tables(["P"], True)
    tables.travel_time_tables = {}
    with pytest.raises(ValueError):
        save_path = os.path.join(tmp_path, "empty_table.npz")
        tables.save(save_path)


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
def test_compute_travel_time(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
    phases: list[str],
):
    """Travel-time interpolation should return the reference values."""
    tables = travel_time_tables(phases, True)
    depth = 50.0
    distance = 500.0
    if "P" in phases:
        travel_time_p = tables.compute_travel_time(distance, depth, "P")
        assert isinstance(travel_time_p, float)
        assert pytest.approx(travel_time_p, rel=1e-3) == 65.54952023370768
    if "S" in phases:
        travel_time_s = tables.compute_travel_time(distance, depth, "S")
        assert isinstance(travel_time_s, float)
        assert pytest.approx(travel_time_s, rel=1e-3) == 117.19924749280987


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
def test_compute_takeoff_angle(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
    phases: list[str],
):
    """Takeoff-angle interpolation should return the reference values."""
    tables = travel_time_tables(phases, True)
    depth = 50.0
    distance = 500.0
    if "P" in phases:
        takeoff_angle_p = tables.compute_takeoff_angle(distance, depth, "P")
        assert isinstance(takeoff_angle_p, float)
        assert pytest.approx(takeoff_angle_p, rel=1e-1) == 89.65376553407343
    if "S" in phases:
        takeoff_angle_s = tables.compute_takeoff_angle(distance, depth, "S")
        assert isinstance(takeoff_angle_s, float)
        assert pytest.approx(takeoff_angle_s, rel=1e-1) == 88.82780282349177


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
def test_get_arrival_1d(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
    phases: list[str],
):
    """1D arrivals should match the expected table values."""
    tables = travel_time_tables(phases, False)
    depth = 50.0
    distance = 500.0
    if "P" in phases:
        time_p, angle_p = tables.get_arrival_1d(distance, depth, "P")
        assert isinstance(time_p, float)
        assert pytest.approx(time_p, rel=1e-3) == 65.54952023370768
        assert isinstance(angle_p, float)
        assert pytest.approx(angle_p, rel=1e-1) == 89.65376553407343
    if "S" in phases:
        time_s, angle_s = tables.get_arrival_1d(distance, depth, "S")
        assert isinstance(time_s, float)
        assert pytest.approx(time_s, rel=1e-3) == 117.19924749280987
        assert isinstance(angle_s, float)
        assert pytest.approx(angle_s, rel=1e-1) == 88.82780282349177


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
@pytest.mark.parametrize("depth,distance", [(150.0, 500.0), (50.0, 1500.0)])
def test_get_arrival_1d_out_of_bounds(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
    phases: list[str],
    depth: float,
    distance: float,
):
    """Out-of-range 1D queries should raise a validation error."""
    tables = travel_time_tables(phases, False)
    if "P" in phases:
        with pytest.raises(ValueError):
            tables.get_arrival_1d(distance, depth, "P")
    if "S" in phases:
        with pytest.raises(ValueError):
            tables.get_arrival_1d(distance, depth, "S")


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
def test_get_arrival_1d_negative_distance(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
    phases: list[str],
):
    """Negative distances should mirror positive-distance 1D arrivals."""
    tables = travel_time_tables(phases, False)
    depth = 50.0
    distance = -500.0
    if "P" in phases:
        time_neg_p, angle_neg_p = tables.get_arrival_1d(distance, depth, "P")
        time_pos_p, angle_pos_p = tables.get_arrival_1d(-distance, depth, "P")
        assert pytest.approx(time_neg_p, rel=1e-3) == time_pos_p
        assert pytest.approx(angle_neg_p, rel=1e-1) == angle_pos_p
    if "S" in phases:
        time_neg_s, angle_neg_s = tables.get_arrival_1d(distance, depth, "S")
        time_pos_s, angle_pos_s = tables.get_arrival_1d(-distance, depth, "S")
        assert pytest.approx(time_neg_s, rel=1e-3) == time_pos_s
        assert pytest.approx(angle_neg_s, rel=1e-1) == angle_pos_s


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
def test_invalid_phase_1d(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
    phases: list[str],
):
    """Invalid 1D phases should raise a validation error."""
    tables = travel_time_tables(phases, False)
    depth = 50.0
    distance = 500.0
    with pytest.raises(ValueError):
        tables.get_arrival_1d(distance, depth, "X")


@pytest.mark.parametrize("phases", [["P"], ["S"], ["P", "S"]])
def test_get_arrival(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
    phases: list[str],
):
    """2D arrivals should match the expected geodesic reference values."""
    tables = travel_time_tables(phases, False)
    source_latitude = 0.0
    source_longitude = 0.0
    source_depth_km = 50.0
    receiver_latitude = 0.0
    receiver_longitude = 4.5  # ~500 km at the equator
    receiver_elevation_m = 0.0
    if "P" in phases:
        arc_distance_km_p, time_p, takeoff_angle_p, azimuth_deg_p = tables.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "P",
        )
        assert isinstance(arc_distance_km_p, float)
        assert pytest.approx(arc_distance_km_p, rel=1e-3) == 500.0
        assert isinstance(time_p, float)
        assert pytest.approx(time_p, rel=1e-3) == 65.54952023370768
        assert isinstance(takeoff_angle_p, float)
        assert pytest.approx(takeoff_angle_p, rel=1e-1) == 89.65376553407343
        assert isinstance(azimuth_deg_p, float)
        assert pytest.approx(azimuth_deg_p, rel=1e-1) == 90.0
    if "S" in phases:
        arc_distance_km_s, time_s, takeoff_angle_s, azimuth_deg_s = tables.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "S",
        )
        assert isinstance(arc_distance_km_s, float)
        assert pytest.approx(arc_distance_km_s, rel=1e-3) == 500.0
        assert isinstance(time_s, float)
        assert pytest.approx(time_s, rel=1e-3) == 117.19924749280987
        assert isinstance(takeoff_angle_s, float)
        assert pytest.approx(takeoff_angle_s, rel=1e-1) == 88.82780282349177
        assert isinstance(azimuth_deg_s, float)
        assert pytest.approx(azimuth_deg_s, rel=1e-1) == 90.0


@pytest.mark.parametrize("phases", [["P"], ["S"]])
def test_get_arrival_invalid_phase(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
    phases: list[str],
):
    """Invalid phases should raise a validation error for 2D arrivals."""
    tables = travel_time_tables(phases, False)
    source_latitude = 0.0
    source_longitude = 0.0
    source_depth_km = 50.0
    receiver_latitude = 0.0
    receiver_longitude = 4.5
    receiver_elevation_m = 0.0
    with pytest.raises(ValueError):
        tables.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "X",
        )


@pytest.mark.parametrize("phases", [["P"], ["S"]])
def test_get_arrival_out_of_bounds(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
    phases: list[str],
):
    """Out-of-range depths should raise a validation error for arrivals."""
    tables = travel_time_tables(phases, False)
    source_latitude = 0.0
    source_longitude = 0.0
    source_depth_km = 150.0
    receiver_latitude = 0.0
    receiver_longitude = 4.5
    receiver_elevation_m = 0.0
    if "P" in phases:
        with pytest.raises(ValueError):
            tables.get_arrival(
                source_latitude,
                source_longitude,
                source_depth_km,
                receiver_latitude,
                receiver_longitude,
                receiver_elevation_m,
                "P",
            )
    if "S" in phases:
        with pytest.raises(ValueError):
            tables.get_arrival(
                source_latitude,
                source_longitude,
                source_depth_km,
                receiver_latitude,
                receiver_longitude,
                receiver_elevation_m,
                "S",
            )


@pytest.mark.parametrize("phases", [["P"], ["S"]])
def test_get_arrival_max_distance_exceeded(
    travel_time_tables: Callable[[list[str], bool], TauTravelTimeGrid],
    phases: list[str],
):
    """Queries beyond the configured max distance should short-circuit."""
    tables = travel_time_tables(phases, False)
    source_latitude = 0.0
    source_longitude = 0.0
    source_depth_km = 50.0
    receiver_latitude = 0.0
    receiver_longitude = 10.0  # ~1000 km at the equator
    receiver_elevation_m = 0.0
    max_distance_km = 500.0
    if "P" in phases:
        arc_distance_km_p, time_p, takeoff_angle_p, azimuth_deg_p = tables.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "P",
            max_distance_km=max_distance_km,
        )

        assert time_p is None
        assert takeoff_angle_p is None
        assert isinstance(azimuth_deg_p, float)
        assert pytest.approx(azimuth_deg_p, rel=1e-1) == 90.0
        assert isinstance(arc_distance_km_p, float)
        assert pytest.approx(arc_distance_km_p, rel=1e-3) == 1111.9
    if "S" in phases:
        arc_distance_km_s, time_s, takeoff_angle_s, azimuth_deg_s = tables.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "S",
            max_distance_km=max_distance_km,
        )

        assert time_s is None
        assert takeoff_angle_s is None
        assert isinstance(azimuth_deg_s, float)
        assert pytest.approx(azimuth_deg_s, rel=1e-1) == 90.0
        assert isinstance(arc_distance_km_s, float)
        assert pytest.approx(arc_distance_km_s, rel=1e-3) == 1111.9
