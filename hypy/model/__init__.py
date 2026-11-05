"""Public velocity-model package API.

This module exposes the stable loader/facade used by locator code and
lazily re-exports the common concrete model wrappers to keep import
paths short while avoiding import cycles.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, Union

if TYPE_CHECKING:
    from .layered.tvel import TvelModel
    from .volumetric.netcdf import NCGrid
    from .volumetric.nllgrid import NllGridModel


class VelocityModel:
    """Common base for velocity model wrappers."""

    model_name: str
    model_path: Path
    dim: int
    feature_flags: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        model_path: Union[str, Path],
        model_name: Optional[str] = None,
        dim: Optional[int] = None,
        radius_of_planet_km: float = 6371.0,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        self.model_name = model_name or self.model_path.stem
        self.dim = dim if dim is not None else -1
        self.radius_of_planet_km = radius_of_planet_km


def __getattr__(name: str):
    """Lazily expose common concrete velocity model backends."""
    if name == "TvelModel":
        from .layered.tvel import TvelModel

        return TvelModel
    if name == "NCGrid":
        from .volumetric.netcdf import NCGrid

        return NCGrid
    if name == "NllGridModel":
        from .volumetric.nllgrid import NllGridModel

        return NllGridModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VelocityModel",
    "TvelModel",
    "NCGrid",
    "NllGridModel",
]
