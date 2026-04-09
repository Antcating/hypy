"""Public velocity-model package API.

This module exposes the stable loader/facade used by locator code and lazily
re-exports the common concrete model wrappers to keep import paths short while
avoiding import cycles.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar, Optional, Union

from ..locator.config import VALID_VElOCITY_MODEL_FTYPES

if TYPE_CHECKING:
    from .layered.tvel import TvelModel
    from .volumetric.netcdf import NCGrid
    from .volumetric.nllgrid import NllGridModel


class VelocityModel:
    """Common base for velocity model wrappers."""

    model_name: str
    model_path: Path
    dim: int
    _registry: ClassVar[dict[str, Callable[..., "VelocityModel"]]] = {}

    def __init__(
        self,
        model_path: Union[str, Path],
        model_name: Optional[str] = None,
        dim: Optional[int] = None,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        self.model_name = model_name or self.model_path.stem
        self.dim = dim if dim is not None else -1

    @classmethod
    def register_loader(
        cls, suffix: str, loader: Callable[..., "VelocityModel"]
    ) -> None:
        """Register a loader callable for a model file suffix."""
        cls._registry[suffix.lower()] = loader

    @classmethod
    def _ensure_default_registry(cls) -> None:
        """Populate built-in loaders once, while allowing external overrides."""
        if cls._registry:
            return

        from .layered.tvel import TvelModel
        from .volumetric.netcdf import NCGrid
        from .volumetric.nllgrid import NllGridModel

        cls.register_loader(".tvel", lambda path, **kwargs: TvelModel(path, **kwargs))
        cls.register_loader(".nllgrid", lambda path, **kwargs: NllGridModel(path))
        cls.register_loader(".nc", lambda path, **kwargs: NCGrid(path))

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
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Velocity model file not found: {model_path}")

        suffix = model_path.suffix.lower()
        if suffix not in VALID_VElOCITY_MODEL_FTYPES:
            raise ValueError(
                f"Unsupported velocity model file type: {suffix}. "
                f"Supported types are: {', '.join(VALID_VElOCITY_MODEL_FTYPES)}."
            )

        cls._ensure_default_registry()
        loader = cls._registry.get(suffix)
        if loader is None:
            raise ValueError(
                f"Unsupported velocity model file type: {suffix}. "
                f"Supported types are: {', '.join(VALID_VElOCITY_MODEL_FTYPES)}."
            )

        return loader(model_path, **kwargs)


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


__all__ = ["VelocityModel", "TvelModel", "NCGrid", "NllGridModel"]
