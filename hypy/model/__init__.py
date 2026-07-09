"""Public velocity-model package API.

This module exposes the stable loader/facade used by locator code and lazily
re-exports the common concrete model wrappers to keep import paths short while
avoiding import cycles.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Collection, Optional, Union

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

    @property
    def supported_features(self) -> frozenset[str]:
        """Return model feature flags exposed for compatibility checks."""
        inferred: set[str] = set(self.feature_flags)
        if self.dim > 0:
            inferred.add(f"dimension.{self.dim}d")
        return frozenset(inferred)

    def supports_feature(self, feature: str) -> bool:
        """Return whether this model supports a single feature flag."""
        return feature in self.supported_features

    def supports_all_features(self, features: Collection[str]) -> bool:
        """Return whether this model supports all provided feature flags."""
        return set(features).issubset(self.supported_features)

    @classmethod
    def supported_model_file_types(cls) -> tuple[str, ...]:
        """Return built-in model file suffixes supported by hypy."""
        return (".tvel", ".nllgrid", ".nc")

    @classmethod
    def _resolve_builtin_loader(cls, suffix: str):
        suffix = suffix.lower()
        if suffix == ".tvel":
            from .layered.tvel import TvelModel

            return lambda path, **kwargs: TvelModel(path, **kwargs)
        if suffix == ".nllgrid":
            from .volumetric.nllgrid import NllGridModel

            return lambda path, **kwargs: NllGridModel(path)
        if suffix == ".nc":
            from .volumetric.netcdf import NCGrid

            return lambda path, **kwargs: NCGrid(path)
        return None

    @classmethod
    def create_model(cls, model_path: Union[str, Path], **kwargs) -> "VelocityModel":
        """Load a velocity model from a file path using built-in suffix routing."""
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Velocity model file not found: {model_path}")

        suffix = model_path.suffix.lower()
        loader = cls._resolve_builtin_loader(suffix)
        if loader is None:
            supported_file_types = ", ".join(cls.supported_model_file_types())
            raise ValueError(
                f"Unsupported velocity model file type: {suffix!r}. "
                f"Supported types are: {supported_file_types}."
            )

        return loader(model_path, **kwargs)

    @classmethod
    def from_file(cls, model_path: Union[str, Path], **kwargs) -> "VelocityModel":
        """Load a velocity model from a file."""
        return cls.create_model(model_path, **kwargs)


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
