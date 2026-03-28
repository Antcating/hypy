import os
from pathlib import Path

import obspy  # type: ignore[import-untyped]

from hypy.propagation.volumetric.grid.pyekfmm import PyekfmmTimeTables


def _iasp91_tvel_path() -> str:
    obspy_path = Path(obspy.__file__).parents[0].as_posix()
    return os.path.join(obspy_path, "taup/data/iasp91.tvel")


def test_pyekfmm_arrival_from_taup_nll_grid():
    model_tvel = _iasp91_tvel_path()

    max_distance_km = 200.0
    distance_step_km = 1.0
    max_depth_km = 50.0
    depth_step_km = 1.0

    tracer = PyekfmmTimeTables(
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
