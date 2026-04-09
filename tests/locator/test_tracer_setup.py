"""Tests for typed propagation setup wiring in locator."""

from pathlib import Path

import pytest

from hypy.locator.propagation_setup import init_propagation
from hypy.locator.types import PropagationConfig
from hypy.propagation.base import PropagationEngine


class _DummyTracer(PropagationEngine):
    """Minimal tracer used for init wiring assertions."""

    @property
    def dim(self) -> int:
        """Return a sentinel dimensionality for tests."""
        return 1

    def get_arrival(self, *args, **kwargs):
        """Return placeholder arrival tuple."""
        return 0.0, 0.0, 0.0, 0.0

    def compute_travel_time_params(self, *args, **kwargs):
        """Return placeholder travel-time parameters tuple."""
        return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)


def test_init_propagation_passes_typed_config_to_factory(monkeypatch):
    """init_propagation should forward fields from PropagationConfig without mutation."""
    captured = {}

    def _fake_create_tracer(**kwargs):
        captured.update(kwargs)
        return _DummyTracer(model_path=kwargs["model_path"].as_posix())

    monkeypatch.setattr(
        "hypy.locator.propagation_setup.create_propagation_backend",
        _fake_create_tracer,
    )

    config = PropagationConfig(
        mode="table-1d",
        engine="taup",
        phases=("P", "S"),
        model_path=Path("/tmp/test-model.tvel"),
        max_depth=100.0,
        max_distance=250.0,
        depth_step=2.0,
        distance_step=5.0,
    )

    tracer = init_propagation(config)

    assert isinstance(tracer, _DummyTracer)
    assert captured["mode"] == config.mode
    assert captured["engine"] == config.engine
    assert captured["phases"] == config.phases
    assert captured["model_path"] == config.model_path
    assert captured["max_depth"] == config.max_depth
    assert captured["max_distance"] == config.max_distance
    assert captured["depth_step"] == config.depth_step
    assert captured["distance_step"] == config.distance_step


def test_init_propagation_legacy_kwargs_path_emits_deprecation(monkeypatch):
    """Legacy init_propagation keyword arguments should still work during migration."""
    captured = {}

    def _fake_create_tracer(**kwargs):
        captured.update(kwargs)
        return _DummyTracer(model_path=kwargs["model_path"].as_posix())

    monkeypatch.setattr(
        "hypy.locator.propagation_setup.create_propagation_backend",
        _fake_create_tracer,
    )

    with pytest.deprecated_call(match="legacy keyword arguments"):
        tracer = init_propagation(
            mode="table-1d",
            engine="taup",
            phases=("P",),
            model_path=Path("/tmp/test-model.tvel"),
            max_depth=100.0,
            max_distance=250.0,
            depth_step=2.0,
            distance_step=5.0,
        )

    assert isinstance(tracer, _DummyTracer)
    assert captured["mode"] == "table-1d"
    assert captured["engine"] == "taup"


def test_init_propagation_missing_required_legacy_fields_raises():
    """Legacy path should fail fast when required arguments are missing."""
    with pytest.raises(ValueError, match="Either pass a PropagationConfig"):
        init_propagation(mode="table-1d", engine="taup")
