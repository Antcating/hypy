from pytest import raises

from hypy.model import VelocityModel
from hypy.propagation.base import PropagationEngine


def _custom_propagation_engine(model: VelocityModel) -> PropagationEngine:
    """Create custom propagation engine for testing."""

    class CustomEngine(PropagationEngine):
        feature_flags = frozenset({"propagation.custom"})
        required_model_features = frozenset({"phase.p", "model.profile.1d"})

    return CustomEngine(model)


def _custom_velocity_model(flags: frozenset) -> VelocityModel:
    """Create custom velocity model for testing."""

    class CustomModel(VelocityModel):
        model_name = "CustomModel"
        feature_flags = flags

    return CustomModel("")


def test_velocity_model_validation_valid():
    """Test that compatible velocity models are accepted by the propagation engine."""
    model = _custom_velocity_model(frozenset({"phase.p", "model.profile.1d"}))
    engine = _custom_propagation_engine(model)
    assert engine.validate_model(model) is True


def test_velocity_model_validation_invalid():
    """Test that incompatible velocity models are rejected by the propagation engine."""
    model = _custom_velocity_model(
        frozenset({"model.profile.1d"})
    )  # Missing model.profile.1d

    with raises(ValueError):
        _custom_propagation_engine(model)
