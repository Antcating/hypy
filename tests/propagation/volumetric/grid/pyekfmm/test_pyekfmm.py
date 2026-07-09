"""Tests for pyekfmm-based volumetric travel-time backends."""

import os
from pathlib import Path

import obspy  # type: ignore[import-untyped]

from hypy.propagation.volumetric.grid.pyekfmm import FmmTravelTimeGrid3D


def _iasp91_tvel_path() -> str:
    obspy_path = Path(obspy.__file__).parents[0].as_posix()
    return os.path.join(obspy_path, "taup/data/iasp91.tvel")


def test_pyekfmm_arrival_from_taup_nll_grid():
    """TVEL-backed pyekfmm should return valid first-arrival travel times."""
    model_tvel = _iasp91_tvel_path()

    max_distance_km = 200.0
    distance_step_km = 1.0
    max_depth_km = 50.0
    depth_step_km = 1.0

    tracer = FmmTravelTimeGrid3D(
        model_path=model_tvel,
        max_distance_km=max_distance_km,
        distance_step_km=distance_step_km,
        max_depth_km=max_depth_km,
        depth_step_km=depth_step_km,
        phases=["P", "S"],
    )

    arrival_p = tracer.get_arrival(
        source_latitude=10.0,
        source_longitude=10.0,
        source_depth_km=2.0,
        receiver_latitude=10.1,
        receiver_longitude=10.1,
        receiver_elevation_m=0.0,
        phase="P",
    )
    arrival_s = tracer.get_arrival(
        source_latitude=10.0,
        source_longitude=10.0,
        source_depth_km=2.0,
        receiver_latitude=10.1,
        receiver_longitude=10.1,
        receiver_elevation_m=0.0,
        phase="S",
    )

    arc_distance_km_p, travel_time_p, takeoff_angle_p, dt_dr_p = arrival_p
    arc_distance_km_s, travel_time_s, takeoff_angle_s, dt_dr_s = arrival_s

    assert arc_distance_km_p > 0.0
    assert arc_distance_km_s > 0.0
    assert travel_time_p is not None
    assert travel_time_s is not None
    assert travel_time_p > 0.0
    assert travel_time_s > 0.0
    assert travel_time_s > travel_time_p
    assert takeoff_angle_p is None
    assert takeoff_angle_s is None
    assert dt_dr_p == 0.0
    assert dt_dr_s == 0.0


def test_pyekfmm_arrival_from_netcdf_nc_grid():
    """NetCDF-backed pyekfmm should return a consistent arrival tuple."""
    model_nc = Path(__file__).parents[5] / "3d_Vp_Vs_VpVs_models.nc"

    tracer = FmmTravelTimeGrid3D(
        model_path=model_nc.as_posix(),
        max_distance_km=20.0,
        distance_step_km=2.0,
        max_depth_km=10.0,
        depth_step_km=1.0,
        phases=["P"],
    )

    arrival = tracer.get_arrival(
        source_latitude=33.45,
        source_longitude=34.78,
        source_depth_km=5.0,
        receiver_latitude=33.43,
        receiver_longitude=34.81,
        receiver_elevation_m=0.0,
        phase="P",
    )

    arc_distance_km, travel_time, takeoff_angle, dt_dr = arrival

    assert arc_distance_km > 0.0
    assert travel_time is None or travel_time > 0.0
    assert takeoff_angle is None
    assert dt_dr == 0.0
