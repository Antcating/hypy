"""Propagation initialization helpers for locator setup."""

from __future__ import annotations

import warnings

from hypy.propagation import PropagationEngine, create_propagation_backend

from .types import PropagationConfig


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

    return create_propagation_backend(
        mode=config.mode,
        engine=config.engine,
        phases=config.phases,
        model_path=config.model_path,
        max_depth=config.max_depth,
        max_distance=config.max_distance,
        depth_step=config.depth_step,
        distance_step=config.distance_step,
    )
