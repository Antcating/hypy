"""Public propagation package API."""

from .base import PropagationEngine

__all__ = ["PropagationEngine"]


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
