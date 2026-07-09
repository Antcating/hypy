import os
from typing import Any

import obspy
import pytest

from hypy.locator import EventLocator
from hypy.model.layered.tvel import TvelModel
from hypy.propagation.layered.ray_tracer.taup import TauPRayTracer


def build_tvel_model():
    """Build a TvelModel for each supported engine."""
    model_name = "iasp91"
    obspy_path = os.path.dirname(obspy.__file__)
    model_path = os.path.join(obspy_path, "taup", "data", f"{model_name}.tvel")
    return TvelModel(model_path)


def test_event_locator_valid_phases():
    tvel_model = build_tvel_model()
    locator = EventLocator(
        propagation_engine=TauPRayTracer(model=tvel_model),
        phases=("p", "s"),
    )
    assert locator.config.phases == ("p", "s")
    assert locator.propagation_backend.phases == ("p", "s")


def test_event_locator_phases_subset_of_propagation_engine_phases():
    tvel_model = build_tvel_model()
    locator = EventLocator(
        propagation_engine=TauPRayTracer(model=tvel_model, phases=("p", "s")),
        phases=("p"),
    )
    assert locator.config.phases == ("p",)
    assert locator.propagation_backend.phases == ("p", "s")


@pytest.mark.parametrize("phases", [None, 1, 2, 3])
def test_event_locator_invalid_type_phases(phases: Any):
    tvel_model = build_tvel_model()
    with pytest.raises(ValueError, match="Phases must be an iterable of strings"):
        EventLocator(
            propagation_engine=TauPRayTracer(model=tvel_model),
            phases=phases,
        )


@pytest.mark.parametrize(
    ["phases", "propagation_engine_phases"],
    [
        (("p", "s", "Ps"), ("P", "S")),
        (("p", "s", "pS"), ("P", "S")),
        (("P", "S"), ("p")),
    ],
)
def test_event_locator_phases_do_not_match_propagation_engine_phases(
    phases: tuple[str, ...],
    propagation_engine_phases: tuple[str, ...],
):
    tvel_model = build_tvel_model()
    with pytest.raises(
        ValueError,
        match="Propagation engine phases .* do not match provided phases .*",
    ):
        EventLocator(
            propagation_engine=TauPRayTracer(
                model=tvel_model,
                phases=propagation_engine_phases,
            ),
            phases=phases,
        )
