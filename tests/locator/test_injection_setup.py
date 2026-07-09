"""Tests for EventLocator constructor-time backend/model injection."""

from pathlib import Path

import pytest

from hypy.locator import EventLocator
from hypy.model import VelocityModel
from hypy.propagation import PropagationEngine


def _iasp91_tvel_path() -> Path:
    import obspy  # type: ignore[import-untyped]

    obspy_path = Path(obspy.__file__).parents[0]
    return obspy_path / "taup" / "data" / "iasp91.tvel"


class _InjectedVelocityModel(VelocityModel):
    feature_flags = frozenset({"model.profile.1d", "format.custom"})

    def __init__(self, model_path: Path | str):
        super().__init__(model_path=model_path, model_name="custom", dim=1)


class _InjectedVelocityModelMissingFeatures(VelocityModel):
    feature_flags = frozenset({"model.grid.3d"})

    def __init__(self, model_path: Path | str):
        super().__init__(model_path=model_path, model_name="custom", dim=3)


class _InjectedBackend(PropagationEngine):
    default_feature_flags = frozenset({"derivative.numerical", "propagation.layered"})
    default_required_model_features = frozenset(
        {"model.profile.1d", "model.profile.1d"}
    )

    @property
    def dim(self) -> int:
        return 1

    def get_arrival(self, *args, **kwargs):
        return 0.0, 0.0, 0.0, 0.0

    def compute_travel_time_params(self, *args, **kwargs):
        return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)


class _NumericalOnlyBackend(PropagationEngine):
    default_feature_flags = frozenset({"derivative.numerical"})
    default_required_model_features = frozenset()

    @property
    def dim(self) -> int:
        return 1

    def get_arrival(self, *args, **kwargs):
        return 0.0, 0.0, 0.0, 0.0

    def compute_travel_time_params(self, *args, **kwargs):
        return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)


def test_event_locator_accepts_injected_model_and_backend_classes():
    locator = EventLocator(
        model_path_legacy=_iasp91_tvel_path().as_posix(),
        mode_legacy="table-1d",
        engine_legacy="taup",
        derivative_method="numerical",
        velocity_model=_InjectedVelocityModel,
        propagation_engine=_InjectedBackend,
    )

    assert isinstance(locator.velocity_model, _InjectedVelocityModel)
    assert isinstance(locator.propagation_backend, _InjectedBackend)


def test_event_locator_rejects_derivative_not_supported_by_injected_backend():
    with pytest.raises(ValueError, match="Unsupported derivative method"):
        EventLocator(
            model_path_legacy=_iasp91_tvel_path().as_posix(),
            mode_legacy="table-1d",
            engine_legacy="taup",
            derivative_method="analytical",
            velocity_model=_InjectedVelocityModel,
            propagation_engine=_NumericalOnlyBackend,
        )


def test_event_locator_rejects_incompatible_injected_model_backend_pair():
    with pytest.raises(ValueError, match="Incompatible model and propagation backend"):
        EventLocator(
            model_path_legacy=_iasp91_tvel_path().as_posix(),
            mode_legacy="table-1d",
            engine_legacy="taup",
            derivative_method="numerical",
            velocity_model=_InjectedVelocityModelMissingFeatures,
            propagation_engine=_InjectedBackend,
        )


def test_event_locator_default_builtin_path_remains_operational():
    locator = EventLocator(
        model_path_legacy=_iasp91_tvel_path().as_posix(),
        mode_legacy="ray-1d",
        engine_legacy="taup",
        derivative_method="analytical",
    )

    assert isinstance(locator.propagation_backend, PropagationEngine)
