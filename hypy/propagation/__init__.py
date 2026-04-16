"""Public propagation package API."""

from .base import PropagationEngine
from .registry import (
    BACKEND_FEATURE_FLAGS,
    REQUIRED_MODEL_FEATURE_FLAGS,
    create_propagation_backend,
    get_propagation_backend_features,
    list_valid_operating_regimes_by_features,
    validate_model_backend_compatibility,
)

__all__ = [
    "PropagationEngine",
    "BACKEND_FEATURE_FLAGS",
    "REQUIRED_MODEL_FEATURE_FLAGS",
    "get_propagation_backend_features",
    "list_valid_operating_regimes_by_features",
    "validate_model_backend_compatibility",
    "create_propagation_backend",
]


def __getattr__(name: str):
    """Lazily expose common propagation base classes."""
    if name == "LayeredPropagationEngine":
        from .layered import LayeredPropagationEngine

        return LayeredPropagationEngine
    if name == "VolumetricPropagationEngine":
        from .volumetric import VolumetricPropagationEngine

        return VolumetricPropagationEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__.extend(["LayeredPropagationEngine", "VolumetricPropagationEngine"])
