"""Volumetric travel-time grid backends."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pyekfmm.grid import FmmTravelTimeGrid3D

__all__ = [
    "FmmTravelTimeGrid3D",
]


def __getattr__(name: str):
    """Lazy load grid backends to avoid unnecessary dependencies."""
    if name == "FmmTravelTimeGrid3D":
        from .pyekfmm.grid import FmmTravelTimeGrid3D

        return FmmTravelTimeGrid3D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
