"""Tests for propagation tracer registry behavior."""

from pathlib import Path

import pytest

from hypy.model import VelocityModel
from hypy.propagation import (
    create_tracer,
    get_tracer_factory,
    list_registered_tracers,
    load_tracer_plugins,
    register_tracer,
    registry,
)
from hypy.propagation.base import PropagationEngine


class DummyPropagationModel(PropagationEngine):
    """Minimal propagation backend used to test registry wiring."""

    @property
    def dim(self) -> int:
        """Return a sentinel dimension for test assertions."""
        return 0

    def get_arrival(self, *args, **kwargs):
        """Return stable placeholder arrival values for tests."""
        return 0.0, 0.0, 0.0, 0.0

    def compute_travel_time_params(self, *args, **kwargs):
        """Return stable placeholder travel-time parameter values for tests."""
        return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)


def test_register_and_create_custom_tracer():
    """A custom registration should be discoverable and instantiable."""
    mode = "unit-test-mode"
    engine = "unit-test-engine"

    def _factory(*, model_path: Path, **kwargs):
        assert kwargs["mode"] == mode
        assert kwargs["engine"] == engine
        return DummyPropagationModel(model_path=model_path.as_posix())

    register_tracer(mode, engine, _factory, override=True)

    tracer = create_tracer(
        mode=mode,
        engine=engine,
        phases=("P",),
        model_path=Path("/tmp/unit-test.tvel"),
        max_depth=None,
        max_distance=None,
        depth_step=None,
        distance_step=None,
    )

    assert isinstance(tracer, DummyPropagationModel)
    assert get_tracer_factory(mode, engine) is not None


def test_register_tracer_duplicate_without_override_raises():
    """Duplicate registration without override should fail fast."""
    mode = "unit-test-mode-duplicate"
    engine = "unit-test-engine-duplicate"

    def _factory(*, model_path: Path, **kwargs):
        return DummyPropagationModel(model_path=model_path.as_posix())

    register_tracer(mode, engine, _factory, override=True)

    with pytest.raises(ValueError, match="already registered"):
        register_tracer(mode, engine, _factory)


def test_create_tracer_invalid_mode_raises():
    """Unknown modes should raise a clear unsupported-mode error."""
    with pytest.raises(ValueError, match="Unsupported mode"):
        create_tracer(
            mode="this-mode-does-not-exist",
            engine="taup",
            phases=("P",),
            model_path=Path("/tmp/unit-test.tvel"),
            max_depth=None,
            max_distance=None,
            depth_step=None,
            distance_step=None,
        )


def test_default_registry_contains_expected_builtin_pairs():
    """Built-in mode/engine pairs should be registered on first creation attempt."""
    with pytest.raises(ValueError, match="Unsupported mode"):
        create_tracer(
            mode="trigger-default-registry",
            engine="taup",
            phases=("P",),
            model_path=Path("/tmp/unit-test.tvel"),
            max_depth=None,
            max_distance=None,
            depth_step=None,
            distance_step=None,
        )

    keys = set(list_registered_tracers())
    assert ("table-1d", "fmm") in keys
    assert ("table-1d", "taup") in keys
    assert ("table-1d", "pyrocko") in keys
    assert ("ray-1d", "taup") in keys
    assert ("ray-1d", "pyrocko") in keys
    assert ("table-3d", "fmm") in keys


def test_create_tracer_unsupported_engine_for_valid_mode_raises():
    """Valid modes with unknown engines should return an engine-specific error."""
    with pytest.raises(ValueError, match="Unsupported engine"):
        create_tracer(
            mode="table-1d",
            engine="this-engine-does-not-exist",
            phases=("P",),
            model_path=Path("/tmp/unit-test.tvel"),
            max_depth=None,
            max_distance=None,
            depth_step=None,
            distance_step=None,
        )


def test_load_tracer_plugins_registers_valid_entry_points(
    monkeypatch: pytest.MonkeyPatch,
):
    """Plugin loader should register callables from valid mode:engine entry points."""

    class _FakeEntryPoint:
        def __init__(self, name: str, factory):
            self.name = name
            self._factory = factory

        def load(self):
            return self._factory

    def _plugin_factory(*, model_path: Path, **kwargs):
        del kwargs
        return DummyPropagationModel(model_path=model_path.as_posix())

    monkeypatch.setattr(
        registry,
        "_iter_plugin_entry_points",
        lambda: [_FakeEntryPoint("plugin-mode:plugin-engine", _plugin_factory)],
    )
    monkeypatch.setattr(registry, "_plugins_loaded", False)

    loaded = load_tracer_plugins(override=True)
    assert ("plugin-mode", "plugin-engine") in loaded
    assert get_tracer_factory("plugin-mode", "plugin-engine") is not None


def test_create_tracer_rejects_incompatible_velocity_model(monkeypatch):
    """Factory creation should fail for incompatible model feature flags."""
    mode = "compat-mode"
    engine = "compat-engine"

    class _ModelWithoutNeededFeatures(VelocityModel):
        feature_flags = frozenset({"dimension.3d"})

    def _factory(*, model_path: Path, **kwargs):
        del kwargs
        return DummyPropagationModel(model_path=model_path.as_posix())

    register_tracer(mode, engine, _factory, override=True)
    monkeypatch.setitem(
        registry.REQUIRED_MODEL_FEATURE_FLAGS,
        (mode, engine),
        frozenset({"dimension.1d"}),
    )

    with pytest.raises(ValueError, match="Incompatible model and propagation backend"):
        create_tracer(
            mode=mode,
            engine=engine,
            phases=("P",),
            model_path=Path("/tmp/unit-test.tvel"),
            max_depth=None,
            max_distance=None,
            depth_step=None,
            distance_step=None,
            velocity_model=_ModelWithoutNeededFeatures("/tmp/unit-test-model.nc"),
        )
