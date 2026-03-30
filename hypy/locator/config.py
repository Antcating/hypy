from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .types import EngineLiteral, ModeLiteral, SolverLiteral, DerivativeMethodLiteral


VALID_OPERATING_REGIMES = {
    "ray": {
        "taup": {"analytical", "numerical"},
        "pyrocko": {"analytical", "numerical"},
    },
    "table": {
        "taup": {"analytical", "numerical"},
        "pyrocko": {"analytical", "numerical"},
        "fmm": {"numerical"},
    },
}

VALID_SOLVERS = {"lsqr", "svd"}


def validate_operating_regime(
    mode: str, engine: str
) -> "tuple[ModeLiteral, EngineLiteral]":
    if mode not in VALID_OPERATING_REGIMES:
        raise ValueError(
            f"Unsupported mode: {mode!r}. Use one of: {', '.join(VALID_OPERATING_REGIMES)}."
        )
    if engine not in VALID_OPERATING_REGIMES[mode]:
        valid_engines = ", ".join(VALID_OPERATING_REGIMES[mode])
        raise ValueError(
            f"Unsupported engine: {engine!r} for mode {mode!r}. "
            f"Valid engines for this mode are: {valid_engines}."
        )
    return mode, engine  # type: ignore[return-value]


def validate_solver(solver: str) -> SolverLiteral:
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
    """
    Compute logistic weighting parameters (k, x0) if all inputs are provided.
    Returns: (enabled, k, x0).
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
    max_depth: Optional[float], depth_step: Optional[float]
) -> None:
    if max_depth is None or depth_step is None:
        raise ValueError(
            "When using precalculated velocities, 'max_depth' and 'depth_step' must be provided."
        )


@dataclass(frozen=True)
class LocatorConfig:
    """
    Immutable configuration for the event locator.

    Holds all settings that do not change during a locate() run.
    Created once in EventLocator.__init__ and shared (read-only) with all components.
    """

    tvel_file_path: Path
    model_name: str
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
        return len(self.phases)
