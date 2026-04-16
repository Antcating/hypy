"""Propagation initialization helpers for locator setup."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import cast

from hypy.model import VelocityModel
from hypy.propagation import (
    BACKEND_FEATURE_FLAGS,
    REQUIRED_MODEL_FEATURE_FLAGS,
    PropagationEngine,
    create_propagation_backend,
)

from .types import PropagationBackendType, PropagationConfig


def resolve_backend_contract(
    mode: str,
    engine: str,
    engine_backend: PropagationBackendType | None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Resolve propagation feature contracts from built-ins or injected backend."""
    if engine_backend is None:
        return (
            BACKEND_FEATURE_FLAGS.get((mode, engine), frozenset()),
            REQUIRED_MODEL_FEATURE_FLAGS.get((mode, engine), frozenset()),
        )

    feature_flags: frozenset[str] = getattr(
        engine_backend,
        "default_feature_flags",
        frozenset(),
    )
    required_features: frozenset[str] = getattr(
        engine_backend,
        "default_required_model_features",
        frozenset(),
    )
    return frozenset(feature_flags), frozenset(required_features)


def _instantiate_injected_backend(
    engine_backend: PropagationBackendType,
    config: PropagationConfig,
) -> PropagationEngine:
    model_path = Path(config.model_path).as_posix()

    attempts: list[dict[str, object]] = [
        {
            "model_path": model_path,
            "mode": config.mode,
            "engine": config.engine,
            "phases": list(config.phases),
            "max_depth_km": config.max_depth,
            "max_distance_km": config.max_distance,
            "depth_step_km": config.depth_step,
            "distance_step_km": config.distance_step,
        },
        {
            "model_path": model_path,
            "mode": config.mode,
            "engine": config.engine,
            "phases": list(config.phases),
            "max_depth": config.max_depth,
            "max_distance": config.max_distance,
            "depth_step": config.depth_step,
            "distance_step": config.distance_step,
        },
        {
            "model_path": model_path,
        },
    ]

    backend: PropagationEngine | None = None
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            backend = engine_backend(**cast(dict[str, object], kwargs))  # type: ignore[arg-type]
            break
        except TypeError as exc:
            last_error = exc

    if backend is None:
        raise TypeError(
            "Unable to initialize engine_backend with compatible constructor. "
            "Expected a class compatible with hypy propagation backends."
        ) from last_error

    backend.mode = config.mode
    backend.engine = config.engine
    return backend


def init_propagation(
    config: PropagationConfig | None = None,
    *,
    mode: str | None = None,
    engine: str | None = None,
    phases: tuple[str, ...] | None = None,
    model_path=None,
    max_depth: float | None = None,
    max_distance: float | None = None,
    depth_step: float | None = None,
    distance_step: float | None = None,
    velocity_model: VelocityModel | None = None,
    engine_backend: PropagationBackendType | None = None,
    backend_feature_flags: frozenset[str] | None = None,
    required_model_features: frozenset[str] | None = None,
) -> PropagationEngine:
    """Initialize a propagation backend from typed config or legacy kwargs.

    Parameters are accepted in two forms during migration:

        * Preferred: pass a ``PropagationConfig`` instance as ``config``.
        * Legacy: pass named keyword arguments (mode/engine/phases/model_path/etc.).
            Legacy usage emits a deprecation warning.
    """
    if config is None:
        if mode is None or engine is None or phases is None or model_path is None:
            raise ValueError(
                "Either pass a PropagationConfig instance or provide all legacy "
                "propagation arguments: mode, engine, phases, model_path."
            )
        warnings.warn(
            "Using init_propagation with legacy keyword arguments is deprecated; "
            "pass PropagationConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        config = PropagationConfig(
            mode=mode,  # type: ignore[arg-type]
            engine=engine,  # type: ignore[arg-type]
            phases=phases,
            model_path=model_path,
            max_depth=max_depth,
            max_distance=max_distance,
            depth_step=depth_step,
            distance_step=distance_step,
        )

    if engine_backend is None:
        return create_propagation_backend(
            mode=config.mode,
            engine=config.engine,
            phases=config.phases,
            model_path=config.model_path,
            max_depth=config.max_depth,
            max_distance=config.max_distance,
            depth_step=config.depth_step,
            distance_step=config.distance_step,
            velocity_model=velocity_model,
        )

    backend = _instantiate_injected_backend(engine_backend, config)
    backend.feature_flags = frozenset(
        backend_feature_flags
        if backend_feature_flags is not None
        else getattr(engine_backend, "default_feature_flags", frozenset())
    )
    backend.required_model_features = frozenset(
        required_model_features
        if required_model_features is not None
        else getattr(engine_backend, "default_required_model_features", frozenset())
    )
    return backend
