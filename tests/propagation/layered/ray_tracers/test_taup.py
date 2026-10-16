import os
from pathlib import Path
from types import SimpleNamespace

import obspy
import pytest

from hypy.model.layered.tvel import TvelModel
from hypy.propagation.layered.ray_tracer.taup import TauPRayTracer


def _giimod_tvel_path():
    obspy_path = Path(obspy.__file__).parents[0].as_posix()
    return os.path.join(obspy_path, "taup/data/iasp91.tvel")


def test_taup_ray_tracer_valid_phases():
    tvel_model = TvelModel(_giimod_tvel_path())
    taup_ray_tracer = TauPRayTracer(model=tvel_model)
    assert taup_ray_tracer.phases == ("p", "s")


def test_taup_ray_tracer_invalid_phases():
    with pytest.raises(ValueError):
        TauPRayTracer(model=TvelModel(_giimod_tvel_path()), phases=("P", "S", "X"))


@pytest.mark.parametrize(("phase", "group"), [("P", "ttp"), ("S", "tts")])
def test_taup_uses_phase_family(monkeypatch, phase, group):
    tracer = TauPRayTracer(model=TvelModel(_giimod_tvel_path()))
    called = {}

    def get_travel_times(**kwargs):
        called.update(kwargs)
        return [SimpleNamespace(time=1.0, takeoff_angle=2.0)]

    monkeypatch.setattr(tracer.model, "get_travel_times", get_travel_times)

    assert tracer.get_arrival_1d(10.0, 5.0, phase) == (1.0, 2.0)
    assert called["phase_list"] == [group]
