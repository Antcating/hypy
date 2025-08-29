import os
import obspy
import pytest
from hypy.propagation.ray_tracers.taup import TauPRayTracer


@pytest.fixture
def tracer():
    model = "iasp91"
    obspy_path = os.path.dirname(obspy.__file__)
    model_path = os.path.join(obspy_path, "taup", "data", f"{model}.tvel")
    return TauPRayTracer(model_path=model_path)


def test_get_arrival(tracer):
    depth = 50.0
    distance = 500.0
    time_p, angle_p = tracer.get_arrival(depth, distance, "P")
    time_s, angle_s = tracer.get_arrival(depth, distance, "S")

    assert isinstance(time_p, float)
    assert pytest.approx(time_p, rel=1e-3) == 65.54952023370768
    assert isinstance(angle_p, float)
    assert pytest.approx(angle_p, rel=1e-1) == 89.6292410507832

    assert isinstance(time_s, float)
    assert pytest.approx(time_s, rel=1e-3) == 117.19924749280987
    assert isinstance(angle_s, float)
    assert pytest.approx(angle_s, rel=1e-1) == 88.83770931419198


def test_invalid_phase(tracer):
    depth = 50.0
    distance = 500.0
    with pytest.raises(ValueError):
        tracer.get_arrival(depth, distance, "X")
