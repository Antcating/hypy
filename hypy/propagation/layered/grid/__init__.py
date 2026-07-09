"""Layered travel-time grid backends."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fmm.grid import FmmTravelTimeGrid1D
    from .tau.grid import TauTravelTimeGrid

__all__ = [
    "FmmTravelTimeGrid1D",
    "TauTravelTimeGrid",
]


def __getattr__(name: str):
    """Lazy load grid backends to avoid unnecessary dependencies."""
    if name == "FmmTravelTimeGrid1D":
        from .fmm.grid import FmmTravelTimeGrid1D

        return FmmTravelTimeGrid1D
    elif name == "TauTravelTimeGrid":
        from .tau.grid import TauTravelTimeGrid

        return TauTravelTimeGrid
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
