import numpy as np

from .types import Literal, Optional

ModeLiteral = Literal["table", "ray"]
EngineLiteral = Literal["taup", "pyrocko"]


def validate_mode(mode: str) -> ModeLiteral:
    if mode not in ("table", "ray"):
        raise ValueError(f"Unsupported mode: {mode!r}. Use 'table' or 'ray'.")
    return mode  # type: ignore[return-value]


def validate_engine(engine: str) -> EngineLiteral:
    if engine not in ("taup", "pyrocko"):
        raise ValueError(f"Unsupported engine: {engine!r}. Use 'taup' or 'pyrocko'.")
    return engine  # type: ignore[return-value]


def configure_distance_weighting(
    distance_weight_start: Optional[float],
    distance_weight_end: Optional[float],
    weight_start: Optional[float],
    weight_end: Optional[float],
) -> tuple[bool, Optional[float], Optional[float]]:
    """
    Compute logistic weighting parameters (k, x0) if all inputs are provided.
    Returns: (enabled, k, x0).
    """
    args = (distance_weight_start, distance_weight_end, weight_start, weight_end)
    if any(a is None for a in args):
        return False, None, None

    d0 = float(distance_weight_start)  # type: ignore[arg-type]
    d1 = float(distance_weight_end)  # type: ignore[arg-type]
    w0 = float(weight_start)  # type: ignore[arg-type]
    w1 = float(weight_end)  # type: ignore[arg-type]

    if not (0.0 < w0 < 1.0 and 0.0 < w1 < 1.0):
        raise ValueError("weight_start and weight_end must be in (0, 1).")
    if d1 <= d0:
        raise ValueError(
            "distance_weight_end must be greater than distance_weight_start."
        )

    # Solve logistic parameters for: w(d) = 1 / (1 + exp(-k (d - x0)))
    k = (np.log((1.0 / w1) - 1.0) - np.log((1.0 / w0) - 1.0)) / (d1 - d0)
    x0 = d0 - (np.log((1.0 / w0) - 1.0) / k)
    return True, float(k), float(x0)


def validate_precompute_args(max_depth, depth_step) -> None:
    if max_depth is None or depth_step is None:
        raise ValueError(
            "When using precalculated velocities, 'max_depth' and 'depth_step' must be provided."
        )
