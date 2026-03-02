import os
import obspy
import pytest
from hypy.propagation.ray_tracers.taup import TauPRayTracer
from hypy.propagation.ray_tracers.pyrocko import PyrockoRayTracer
from hypy.propagation.grid.isotropic import TauTimeTables


@pytest.fixture(params=["taup", "pyrocko"])
def tracer_pair(request):
    model = "iasp91"
    obspy_path = os.path.dirname(obspy.__file__)
    model_path = os.path.join(obspy_path, "taup", "data", f"{model}.tvel")
    # Ray tracer
    if request.param == "taup":
        ray_tracer = TauPRayTracer(model_path=model_path)
    else:
        ray_tracer = PyrockoRayTracer(model_path=model_path)
    # Table tracer
    table_tracer = TauTimeTables(
        model_tvel=model_path,
        model_name=model,
        phases=["P", "S"],
        max_depth=100.0,
        max_distance=1000.0,
        depth_step=10.0,
        distance_step=50.0,
        engine=request.param,
        overwrite=True,
    )
    return ray_tracer, table_tracer


@pytest.mark.parametrize("phase", ["P", "S"])
@pytest.mark.parametrize("depth,distance", [(50.0, 500.0), (10.0, 100.0)])
def test_tracer_result_consistency(tracer_pair, phase, depth, distance):
    ray_tracer, table_tracer = tracer_pair
    # Ray tracer
    time_ray, angle_ray = ray_tracer.get_arrival(depth, distance, phase)
    # Table tracer
    time_table, angle_table = table_tracer.get_arrival(depth, distance, phase)
    # Compare travel times and angles
    assert pytest.approx(time_ray, rel=1e-2) == time_table
    assert pytest.approx(angle_ray, rel=1e-1) == angle_table
