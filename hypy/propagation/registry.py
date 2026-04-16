"""Built-in propagation backend registration and compatibility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from hypy.model import VelocityModel

from .base import PropagationEngine

PropagationBackendFactory = Callable[..., PropagationEngine]
ModeEngineKey = tuple[str, str]

# Built-in backend feature registry (avoid costly dynamic inspection at runtime)
BACKEND_FEATURE_FLAGS: dict[ModeEngineKey, frozenset[str]] = {
    ("ray-1d", "taup"): frozenset(
        {
            "propagation.layered",
            "propagation.ray",
            "derivative.analytical",
            "derivative.numerical",
        }
    ),
    ("ray-1d", "pyrocko"): frozenset(
        {
            "propagation.layered",
            "propagation.ray",
            "derivative.analytical",
            "derivative.numerical",
        }
    ),
    ("table-1d", "taup"): frozenset(
        {
            "propagation.layered",
            "propagation.table",
            "derivative.analytical",
            "derivative.numerical",
        }
    ),
    ("table-1d", "pyrocko"): frozenset(
        {
            "propagation.layered",
            "propagation.table",
            "derivative.analytical",
            "derivative.numerical",
        }
    ),
    ("table-1d", "fmm"): frozenset(
        {
            "propagation.layered",
            "propagation.table",
            "derivative.numerical",
        }
    ),
    ("table-3d", "fmm"): frozenset(
        {
            "propagation.volumetric",
            "propagation.table",
            "derivative.numerical",
        }
    ),
}

REQUIRED_MODEL_FEATURE_FLAGS: dict[ModeEngineKey, frozenset[str]] = {
    ("ray-1d", "taup"): frozenset({"model.profile.1d", "dimension.1d"}),
    ("ray-1d", "pyrocko"): frozenset({"model.profile.1d", "dimension.1d"}),
    ("table-1d", "taup"): frozenset({"model.profile.1d", "dimension.1d"}),
    ("table-1d", "pyrocko"): frozenset({"model.profile.1d", "dimension.1d"}),
    ("table-1d", "fmm"): frozenset({"model.profile.1d", "dimension.1d"}),
    ("table-3d", "fmm"): frozenset({"model.grid.3d", "dimension.3d"}),
}


def _require_table_grid_parameters(
    max_depth: Optional[float],
    max_distance: Optional[float],
    depth_step: Optional[float],
    distance_step: Optional[float],
) -> tuple[float, float, float, float]:
    """Normalize and validate table grid generation arguments."""
    if any(
        value is None for value in (max_depth, max_distance, depth_step, distance_step)
    ):
        raise ValueError(
            "In table mode these variables are required: "
            "max_depth, max_distance, depth_step, distance_step"
        )

    assert max_depth is not None
    assert max_distance is not None
    assert depth_step is not None
    assert distance_step is not None

    return (
        float(max_depth),
        float(max_distance),
        float(depth_step),
        float(distance_step),
    )


def _extract_derivative_methods(feature_flags: frozenset[str]) -> set[str]:
    return {
        flag.split(".", 1)[1]
        for flag in feature_flags
        if flag.startswith("derivative.") and "." in flag
    }


def _create_table_1d_fmm(
    *,
    phases: tuple[str, ...],
    model_path: Path,
    max_depth: Optional[float],
    max_distance: Optional[float],
    depth_step: Optional[float],
    distance_step: Optional[float],
    **_: object,
) -> PropagationEngine:
    from hypy.propagation.layered.grid import FmmTravelTimeGrid1D

    max_depth_km, max_distance_km, depth_step_km, distance_step_km = (
        _require_table_grid_parameters(
            max_depth=max_depth,
            max_distance=max_distance,
            depth_step=depth_step,
            distance_step=distance_step,
        )
    )

    return FmmTravelTimeGrid1D(
        model_tvel=model_path.as_posix(),
        phases=list(phases),
        max_depth=max_depth_km,
        max_distance=max_distance_km,
        depth_step=depth_step_km,
        distance_step=distance_step_km,
    )


def _create_table_1d_taupy(
    *,
    engine: str,
    phases: tuple[str, ...],
    model_path: Path,
    max_depth: Optional[float],
    max_distance: Optional[float],
    depth_step: Optional[float],
    distance_step: Optional[float],
    **_: object,
) -> PropagationEngine:
    from hypy.propagation.layered.grid import TauTravelTimeGrid

    max_depth_km, max_distance_km, depth_step_km, distance_step_km = (
        _require_table_grid_parameters(
            max_depth=max_depth,
            max_distance=max_distance,
            depth_step=depth_step,
            distance_step=distance_step,
        )
    )

    return TauTravelTimeGrid(
        model_tvel=model_path.as_posix(),
        phases=list(phases),
        max_depth=max_depth_km,
        max_distance=max_distance_km,
        depth_step=depth_step_km,
        distance_step=distance_step_km,
        engine=engine,
    )


def _create_table_3d_fmm(
    *,
    phases: tuple[str, ...],
    model_path: Path,
    max_depth: Optional[float],
    max_distance: Optional[float],
    depth_step: Optional[float],
    distance_step: Optional[float],
    **_: object,
) -> PropagationEngine:
    from hypy.propagation.volumetric.grid.pyekfmm import FmmTravelTimeGrid3D

    max_depth_km, max_distance_km, depth_step_km, distance_step_km = (
        _require_table_grid_parameters(
            max_depth=max_depth,
            max_distance=max_distance,
            depth_step=depth_step,
            distance_step=distance_step,
        )
    )

    return FmmTravelTimeGrid3D(
        model_path=model_path.as_posix(),
        phases=list(phases),
        max_depth_km=max_depth_km,
        max_distance_km=max_distance_km,
        depth_step_km=depth_step_km,
        distance_step_km=distance_step_km,
    )


def _create_ray_1d_taup(*, model_path: Path, **_: object) -> PropagationEngine:
    from hypy.propagation.layered.ray_tracer.taup import TauPRayTracer

    return TauPRayTracer(model_path=model_path.as_posix())


def _create_ray_1d_pyrocko(*, model_path: Path, **_: object) -> PropagationEngine:
    from hypy.propagation.layered.ray_tracer.pyrocko import PyrockoRayTracer

    return PyrockoRayTracer(model_path=model_path.as_posix())


BACKEND_FACTORY_REGISTRY: dict[ModeEngineKey, PropagationBackendFactory] = {
    ("table-1d", "fmm"): _create_table_1d_fmm,
    ("table-1d", "taup"): _create_table_1d_taupy,
    ("table-1d", "pyrocko"): _create_table_1d_taupy,
    ("table-3d", "fmm"): _create_table_3d_fmm,
    ("ray-1d", "taup"): _create_ray_1d_taup,
    ("ray-1d", "pyrocko"): _create_ray_1d_pyrocko,
}


def get_propagation_backend_features(mode: str, engine: str) -> frozenset[str]:
    """Return feature flags declared for a propagation mode/engine pair."""
    return BACKEND_FEATURE_FLAGS.get((mode, engine), frozenset())


def list_valid_operating_regimes_by_features() -> dict[str, dict[str, set[str]]]:
    """Return mode/engine/derivative combinations inferred from feature flags."""
    regimes: dict[str, dict[str, set[str]]] = {}
    for (mode, engine), features in BACKEND_FEATURE_FLAGS.items():
        derivative_methods = _extract_derivative_methods(features)
        if not derivative_methods:
            continue

        regimes.setdefault(mode, {})[engine] = derivative_methods

    return regimes


def _resolve_builtin_backend_factory(
    mode: str,
    engine: str,
) -> Optional[PropagationBackendFactory]:
    return BACKEND_FACTORY_REGISTRY.get((mode, engine))


def validate_model_backend_compatibility(
    *,
    model_instance: VelocityModel,
    required_model_features: frozenset[str],
    mode: str,
    engine: str,
) -> None:
    """Raise if a model is incompatible with backend required features."""
    if not required_model_features:
        return

    missing_features = sorted(
        required_model_features.difference(model_instance.supported_features)
    )
    if not missing_features:
        return

    raise ValueError(
        "Incompatible model and propagation backend: "
        f"mode={mode!r}, engine={engine!r} requires model features "
        f"{missing_features}, but model {model_instance.model_name!r} "
        f"provides {sorted(model_instance.supported_features)}."
    )


def create_propagation_backend(
    *,
    mode: str,
    engine: str,
    phases: tuple[str, ...],
    model_path: Path,
    max_depth: Optional[float],
    max_distance: Optional[float],
    depth_step: Optional[float],
    distance_step: Optional[float],
    velocity_model: VelocityModel | None = None,
) -> PropagationEngine:
    """Construct the requested built-in propagation backend."""
    backend_factory = _resolve_builtin_backend_factory(mode, engine)
    if backend_factory is None:
        supported_modes = {
            registered_mode for registered_mode, _ in BACKEND_FEATURE_FLAGS
        }
        if mode not in supported_modes:
            raise ValueError(f"Unsupported mode: {mode}")
        raise ValueError(f"Unsupported engine: {engine} for mode {mode}")

    backend = backend_factory(
        mode=mode,
        engine=engine,
        phases=phases,
        model_path=model_path,
        max_depth=max_depth,
        max_distance=max_distance,
        depth_step=depth_step,
        distance_step=distance_step,
    )

    backend.mode = mode
    backend.engine = engine
    backend.feature_flags = get_propagation_backend_features(mode, engine)
    backend.required_model_features = REQUIRED_MODEL_FEATURE_FLAGS.get(
        (mode, engine), frozenset()
    )

    resolved_model = velocity_model or VelocityModel.from_file(model_path)
    validate_model_backend_compatibility(
        model_instance=resolved_model,
        required_model_features=backend.required_model_features,
        mode=mode,
        engine=engine,
    )

    return backend
