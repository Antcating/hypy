import os
from pathlib import Path

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
