"""Public propagation package API.

This module exposes the base propagation model contract and the backend
registration/factory entry points used by locator orchestration code.
"""

from .base import PropagationEngine
from .registry import (
    create_propagation_backend,
    create_tracer,
    get_propagation_backend_factory,
    get_propagation_backend_features,
    get_tracer_factory,
    list_registered_propagation_backends,
    list_registered_tracers,
    list_valid_operating_regimes_by_features,
    load_propagation_plugins,
    load_tracer_plugins,
    register_propagation_backend,
    register_propagation_backend_features,
    register_tracer,
)

__all__ = [
    "PropagationEngine",
    "register_propagation_backend",
    "get_propagation_backend_factory",
    "get_propagation_backend_features",
    "list_registered_propagation_backends",
    "list_valid_operating_regimes_by_features",
    "load_propagation_plugins",
    "create_propagation_backend",
    "register_propagation_backend_features",
    "register_tracer",
    "get_tracer_factory",
    "list_registered_tracers",
    "load_tracer_plugins",
    "create_tracer",
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
