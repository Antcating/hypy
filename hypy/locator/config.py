"""Main configuration and validation logic for the event locator.

This module defines the LocatorConfig dataclass, which holds all the immutable
settings for the event locator. It also includes validation functions to ensure
that the provided configuration parameters are valid and compatible with each other.

Returns
-------
tuple
    A tuple containing:
    - mode: Mode of operation (ModeLiteral).
    - engine: Ray tracing engine (EngineLiteral).
    - derivative_method: Method for computing derivatives (DerivativeMethodLiteral).

Raises
------
ValueError
    If the provided mode is not supported.
ValueError
    If the provided engine is not supported for the specified mode.
ValueError
    If the provided derivative_method is not supported for the mode and engine.
FileNotFoundError
    If a required file is not found.
ValueError
    If a provided file has an unsupported format.
ValueError
    If an unsupported solver is specified.
ValueError
    If distance weighting parameters are invalid.
ValueError
    If precomputation arguments are invalid.
ValueError
    If weight_start or weight_end are not in (0, 1).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from hypy.propagation import list_valid_operating_regimes_by_features

from .types import DerivativeMethodLiteral, EngineLiteral, ModeLiteral, SolverLiteral

VALID_OPERATING_REGIMES = list_valid_operating_regimes_by_features()

VALID_VElOCITY_MODEL_FTYPES = {".tvel", ".nllgrid", ".nc"}

VALID_SOLVERS = {"lsqr", "svd"}


def validate_operating_regime(
    mode: str, engine: str, derivative_method: str
) -> "tuple[ModeLiteral, EngineLiteral, DerivativeMethodLiteral]":
    """Validate that the provided mode and engine are supported, and return them.

    There are specific valid combinations of mode and engine,
    defined in ``VALID_OPERATING_REGIMES``.
    TODO: Add documentation page with a compatibility table showing which
    derivative methods are supported for each mode and engine combination.

    Parameters
    ----------
    mode : str
        Mode of operation. Supported values are ``ray-1d``, ``table-1d``,
        and ``table-3d``.

    engine : str
        Travel-time engine for the selected mode.
        The allowed combinations are defined in ``VALID_OPERATING_REGIMES``.

    derivative_method : str
        Derivative method for the selected mode/engine pair.
        Supported values are ``analytical`` and ``numerical`` depending on the regime.

    Returns
    -------
    tuple of (mode, engine, derivative_method).

    Raises
    ------
    ValueError if the mode, engine, or derivative method is not valid or not compatible.
    """
    valid_operating_regimes = list_valid_operating_regimes_by_features()
    print(f"Valid operating regimes: {valid_operating_regimes}")

    if mode not in valid_operating_regimes:
        raise ValueError(
            f"Unsupported mode: {mode!r}. "
            f"Use one of: {', '.join(valid_operating_regimes)}."
        )
    if engine not in valid_operating_regimes[mode]:
        valid_engines = ", ".join(valid_operating_regimes[mode])
        raise ValueError(
            f"Unsupported engine: {engine!r} for mode {mode!r}. "
            f"Valid engines for this mode are: {valid_engines}."
        )
    if derivative_method not in valid_operating_regimes[mode][engine]:
        valid_methods = ", ".join(valid_operating_regimes[mode][engine])
        raise ValueError(
            f"Unsupported derivative method: "
            f"{derivative_method!r} for mode {mode!r} and engine {engine!r}. "
            f"Valid derivative methods for this mode and engine are: {valid_methods}."
        )
    return mode, engine, derivative_method  # type: ignore[return-value]


def validate_velocity_model_file(model_path: Path) -> Path:
    """Validate that the provided velocity model file exists and has a supported format.

    Supported formats are defined in ``VALID_VElOCITY_MODEL_FTYPES``.

    Parameters
    ----------
    model_path : Path
        Path to the velocity model file.

    Returns
    -------
    Path
        The validated path to the velocity model file.

    Raises
    ------
    FileNotFoundError if the file does not exist.
    ValueError if the file format is not supported.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Velocity model file not found: {model_path}")
    from hypy.model import VelocityModel

    VelocityModel._ensure_default_registry()
    VelocityModel.load_model_plugins()
    valid_model_file_types = set(VelocityModel.list_registered_model_backends())
    if model_path.suffix.lower() not in valid_model_file_types:
        valid_formats = ", ".join(sorted(valid_model_file_types))
        raise ValueError(
            f"Unsupported velocity model file type: {model_path.suffix!r}. "
            f"Supported formats are: {valid_formats}."
        )
    return model_path


def validate_solver(solver: str) -> SolverLiteral:
    """Validate that the provided solver is supported, and return it.

    There are specific valid solvers, defined in VALID_SOLVERS:

    * ``lsqr`` uses ``scipy.sparse.linalg.lsmr`` for large sparse systems.
    * ``svd`` uses ``scipy.linalg.svd`` and is generally less scalable.

    Parameters
    ----------
    solver : str
        solver to use for the linear system in the location inversion.

    Returns
    -------
    SolverLiteral
        The validated solver.

    Raises
    ------
    ValueError
        If the provided solver is not supported.
    """
    if solver.lower() not in VALID_SOLVERS:
        raise ValueError(
            f"Unsupported solver: {solver!r}. Use one of: {', '.join(VALID_SOLVERS)}."
        )
    return solver.lower()  # type: ignore[return-value]


def configure_distance_weighting(
    distance_start_km: Optional[float],
    distance_end_km: Optional[float],
    weight_start: Optional[float],
    weight_end: Optional[float],
) -> "tuple[bool, Optional[float], Optional[float]]":
    """Configure distance weighting parameters for the location inversion.

    The distance weighting function is defined as:
    w(d) = 1 / (1 + 10^(k (d - x0)))

    where d is the distance, k controls the steepness of the weighting curve,
    and x0 is the distance at which the weight is 0.5.

    Parameters
    ----------
    distance_start_km : Optional[float]
        Distance (in km) at which the weighting starts to decrease from weight_start.
    distance_end_km : Optional[float]
        Distance (in km) at which the weighting reaches weight_end.
    weight_start : Optional[float]
        Weight at distance_start_km. Must be in (0, 1).
    weight_end : Optional[float]
        Weight at distance_end_km. Must be in (0, 1).

    Returns
    -------
    tuple[bool, Optional[float], Optional[float]]
        Returns ``(enabled, k, x0)`` where ``enabled`` is ``True`` only when all
        weighting parameters are provided. If weighting is disabled,
        ``k`` and ``x0`` are ``None``.

    Raises
    ------
    ValueError
        If weight_start or weight_end are not in (0, 1).
    ValueError
        If distance_end_km is not greater than distance_start_km.
    """
    args = (distance_start_km, distance_end_km, weight_start, weight_end)
    if any(a is None for a in args):
        return False, None, None

    d0 = float(distance_start_km)  # type: ignore[arg-type]
    d1 = float(distance_end_km)  # type: ignore[arg-type]
    w0 = float(weight_start)  # type: ignore[arg-type]
    w1 = float(weight_end)  # type: ignore[arg-type]

    if not (0.0 < w0 < 1.0 and 0.0 < w1 < 1.0):
        raise ValueError("weight_start and weight_end must be in (0, 1).")
    if d1 <= d0:
        raise ValueError(
            "distance_weight_end must be greater than distance_weight_start."
        )

    # Solve logistic parameters for: w(d) = 1 / (1 + 10^(k (d - x0)))
    # https://www.desmos.com/calculator/221eudc8kc
    k = (np.log10((1.0 / w1) - 1.0) - np.log10((1.0 / w0) - 1.0)) / (d1 - d0)
    x0 = d0 - (np.log10((1.0 / w0) - 1.0) / k)
    return True, float(k), float(x0)


def validate_precompute_args(
    max_depth_km: Optional[float], depth_step_km: Optional[float]
) -> None:
    """Validate the arguments for precomputing velocities.

    Parameters
    ----------
    max_depth_km : Optional[float]
        Maximum depth (in km) for which to precompute velocities.
    depth_step_km : Optional[float]
        Depth step (in km) for precomputing velocities.

    Raises
    ------
    ValueError
        If either `max_depth_km` or `depth_step_km` is None.
    """
    # TODO: This is a bit of a hack to enforce that these arguments
    # are provided when precomputing velocities.
    # Remove it or refactor it when we have a better way to handle
    # precomputation configuration.
    if max_depth_km is None or depth_step_km is None:
        raise ValueError(
            "When using precalculated velocities, "
            "'max_depth' and 'depth_step' must be provided."
        )


@dataclass(frozen=True)
class LocatorConfig:
    """
    Immutable configuration for the event locator.

    Holds all settings that do not change during a locate() run.
    Created once in EventLocator.__init__ and shared (read-only) with all components.
    """

    phases: tuple
    min_picks_per_event: int
    default_time_error: float
    mode: ModeLiteral
    engine: EngineLiteral
    derivative_method: DerivativeMethodLiteral
    max_depth_km: Optional[float]
    max_distance_km: Optional[float]
    depth_step_km: Optional[float]
    distance_step_km: Optional[float]
    fix_depth_km: Optional[float]
    precalculate_velocities: bool
    distance_weighting_enabled: bool
    k: Optional[float]
    x0: Optional[float]
    derivative_delta_km: float = 0.5
    dim: int = 4

    @property
    def n_phases(self) -> int:
        """Return the number of seismic phases being used in the locator.

        Returns
        -------
        int
            The number of seismic phases.
        """
        return len(self.phases)
