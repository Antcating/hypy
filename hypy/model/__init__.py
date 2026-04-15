"""Public velocity-model package API.

This module exposes the stable loader/facade used by locator code and lazily
re-exports the common concrete model wrappers to keep import paths short while
avoiding import cycles.
"""

from __future__ import annotations

import importlib.metadata as metadata
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Collection,
    Optional,
    Protocol,
    Union,
    cast,
)

if TYPE_CHECKING:
    from .layered.tvel import TvelModel
    from .volumetric.netcdf import NCGrid
    from .volumetric.nllgrid import NllGridModel


class _EntryPointLike(Protocol):
    """Subset of importlib entry point API used by plugin loaders."""

    name: str

    def load(self) -> object:
        """Load and return the object referenced by the entry point."""


class VelocityModel:
    """Common base for velocity model wrappers."""

    model_name: str
    model_path: Path
    dim: int
    feature_flags: ClassVar[frozenset[str]] = frozenset()
    _registry: ClassVar[dict[str, Callable[..., "VelocityModel"]]] = {}
    _plugins_loaded: ClassVar[bool] = False
    ENTRY_POINT_GROUP: ClassVar[str] = "hypy.model.backends"

    def __init__(
        self,
        model_path: Union[str, Path],
        model_name: Optional[str] = None,
        dim: Optional[int] = None,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        self.model_name = model_name or self.model_path.stem
        self.dim = dim if dim is not None else -1

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
    def register_model_backend(
        cls,
        file_type: str,
        loader: Callable[..., "VelocityModel"],
        *,
        override: bool = False,
    ) -> None:
        """Register a model loader callable for a file-type suffix."""
        suffix = file_type.lower()
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        if suffix in cls._registry and not override:
            raise ValueError(f"Model backend already registered for {suffix!r}.")
        cls._registry[suffix] = loader

    @classmethod
    def get_model_backend_factory(
        cls, file_type: str
    ) -> Optional[Callable[..., "VelocityModel"]]:
        """Return the registered model backend factory for a file suffix."""
        suffix = file_type.lower()
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return cls._registry.get(suffix)

    @classmethod
    def list_registered_model_backends(cls) -> tuple[str, ...]:
        """Return registered model file suffixes sorted for introspection."""
        return tuple(sorted(cls._registry))

    @classmethod
    def _iter_plugin_entry_points(cls) -> list[_EntryPointLike]:
        """Return entry points for external model backends across Python versions."""
        try:
            return cast(
                list[_EntryPointLike],
                list(metadata.entry_points(group=cls.ENTRY_POINT_GROUP)),
            )
        except TypeError:
            all_entry_points = metadata.entry_points()
            if hasattr(all_entry_points, "select"):
                return cast(
                    list[_EntryPointLike],
                    list(all_entry_points.select(group=cls.ENTRY_POINT_GROUP)),
                )
            legacy_entry_points = cast(Any, all_entry_points)
            return cast(
                list[_EntryPointLike],
                list(legacy_entry_points.get(cls.ENTRY_POINT_GROUP, [])),
            )

    @classmethod
    def load_model_plugins(cls, *, override: bool = False) -> tuple[str, ...]:
        """Load model backend plugins from entry points.

        Plugins should expose entry points in ``hypy.model.backends`` (or legacy
        ``hypy.model.loaders``) with names equal to file suffixes such as
        ``.foo``.
        """
        if cls._plugins_loaded and not override:
            return ()

        loaded: list[str] = []
        for entry_point in cls._iter_plugin_entry_points():
            file_type = str(getattr(entry_point, "name", "")).strip().lower()
            if not file_type:
                continue
            if not file_type.startswith("."):
                file_type = f".{file_type}"

            loaded_object = entry_point.load()
            if not callable(loaded_object):
                continue
            factory = cast(Callable[..., "VelocityModel"], loaded_object)

            try:
                cls.register_model_backend(file_type, factory, override=override)
            except ValueError:
                continue

            loaded.append(file_type)

        cls._plugins_loaded = True
        return tuple(loaded)

    @classmethod
    def _ensure_default_registry(cls) -> None:
        """Populate built-in loaders once, while allowing external overrides."""
        if cls.get_model_backend_factory(".tvel") is None:
            from .layered.tvel import TvelModel

            cls.register_model_backend(
                ".tvel",
                lambda path, **kwargs: TvelModel(path, **kwargs),
            )
        if cls.get_model_backend_factory(".nllgrid") is None:
            from .volumetric.nllgrid import NllGridModel

            cls.register_model_backend(
                ".nllgrid",
                lambda path, **kwargs: NllGridModel(path),
            )
        if cls.get_model_backend_factory(".nc") is None:
            from .volumetric.netcdf import NCGrid

            cls.register_model_backend(
                ".nc",
                lambda path, **kwargs: NCGrid(path),
            )

    @classmethod
    def create_model(cls, model_path: Union[str, Path], **kwargs) -> "VelocityModel":
        """Load a velocity model from a file path using the model registry."""
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Velocity model file not found: {model_path}")

        cls._ensure_default_registry()
        cls.load_model_plugins()
        suffix = model_path.suffix.lower()
        loader = cls.get_model_backend_factory(suffix)
        if loader is None:
            supported_file_types = ", ".join(cls.list_registered_model_backends())
            raise ValueError(
                f"Unsupported velocity model file type: {suffix}. "
                f"Supported types are: {supported_file_types}."
            )

        return loader(model_path, **kwargs)

    @classmethod
    def from_file(cls, model_path: Union[str, Path], **kwargs) -> "VelocityModel":
        """Load a velocity model from a file.

        Load a velocity model from the specified file path, automatically
        determining the appropriate model class to instantiate based on the file
        extension. The method also validates file existence and supported file
        types, and passes any additional keyword arguments to the model
        constructor as needed.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the velocity model file.
            Supported formats include .tvel, .nllgrid, and .nc.

        Returns
        -------
        VelocityModel
            An instance of a VelocityModel subclass appropriate for the input file type,
            providing methods to query velocity information for travel-time calculations

        Raises
        ------
        FileNotFoundError
            If the specified velocity model file does not exist.
        ValueError
            If the file type is unsupported
            or if required parameters are missing for certain model types.
        ValueError
            If the file type is unsupported
            or if required parameters are missing for certain model types.
        """
        return cls.create_model(model_path, **kwargs)

    register_loader = register_model_backend


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
    "register_model_backend",
    "get_model_backend_factory",
    "list_registered_model_backends",
    "load_model_plugins",
    "create_model",
]
