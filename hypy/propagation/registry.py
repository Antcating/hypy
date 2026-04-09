"""Registry and factory helpers for propagation backends."""

from __future__ import annotations

import importlib.metadata as metadata
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, cast

from hypy.model import VelocityModel

from .base import PropagationEngine

PropagationBackendFactory = Callable[..., PropagationEngine]
_registry: dict[tuple[str, str], PropagationBackendFactory] = {}
ENTRY_POINT_GROUP = "hypy.propagation.backends"
_plugins_loaded = False


class _EntryPointLike(Protocol):
    """Subset of importlib entry point API used by plugin loading."""

    name: str

    def load(self) -> object:
        """Load and return the object referenced by the entry point."""


BACKEND_FEATURE_FLAGS: dict[tuple[str, str], frozenset[str]] = {
    (
        "ray-1d",
        "taup",
    ): frozenset(
        {
            "propagation.layered",
            "propagation.ray",
            "derivative.analytical",
            "derivative.numerical",
        }
    ),
    (
        "ray-1d",
        "pyrocko",
    ): frozenset(
        {
            "propagation.layered",
            "propagation.ray",
            "derivative.analytical",
            "derivative.numerical",
        }
    ),
    (
        "table-1d",
        "taup",
    ): frozenset(
        {
            "propagation.layered",
            "propagation.table",
            "derivative.analytical",
            "derivative.numerical",
        }
    ),
    (
        "table-1d",
        "pyrocko",
    ): frozenset(
        {
            "propagation.layered",
            "propagation.table",
            "derivative.analytical",
            "derivative.numerical",
        }
    ),
    (
        "table-1d",
        "fmm",
    ): frozenset(
        {
            "propagation.layered",
            "propagation.table",
            "derivative.numerical",
        }
    ),
    (
        "table-3d",
        "fmm",
    ): frozenset(
        {
            "propagation.volumetric",
            "propagation.table",
            "derivative.numerical",
        }
    ),
}

REQUIRED_MODEL_FEATURE_FLAGS: dict[tuple[str, str], frozenset[str]] = {
    ("ray-1d", "taup"): frozenset({"model.profile.1d", "dimension.1d"}),
    ("ray-1d", "pyrocko"): frozenset({"model.profile.1d", "dimension.1d"}),
    ("table-1d", "taup"): frozenset({"model.profile.1d", "dimension.1d"}),
    ("table-1d", "pyrocko"): frozenset({"model.profile.1d", "dimension.1d"}),
    ("table-1d", "fmm"): frozenset({"model.profile.1d", "dimension.1d"}),
    ("table-3d", "fmm"): frozenset({"model.grid.3d", "dimension.3d"}),
}


def register_propagation_backend_features(
    mode: str,
    engine: str,
    feature_flags: set[str] | frozenset[str],
    *,
    required_model_features: set[str] | frozenset[str] | None = None,
    override: bool = False,
) -> None:
    """Register feature metadata for a propagation backend.

    Feature metadata is used by configuration validation and compatibility
    checks. Third-party backends should register both factory and feature
    metadata so they participate in operating-regime validation.
    """
    key = (mode, engine)
    if key in BACKEND_FEATURE_FLAGS and not override:
        raise ValueError(
            f"Feature flags already registered for mode={mode!r}, engine={engine!r}."
        )

    BACKEND_FEATURE_FLAGS[key] = frozenset(feature_flags)
    if required_model_features is not None:
        REQUIRED_MODEL_FEATURE_FLAGS[key] = frozenset(required_model_features)


def get_propagation_backend_features(mode: str, engine: str) -> frozenset[str]:
    """Return feature flags declared for a propagation mode/engine pair."""
    return BACKEND_FEATURE_FLAGS.get((mode, engine), frozenset())


def list_valid_operating_regimes_by_features() -> dict[str, dict[str, set[str]]]:
    """Return mode/engine/derivative combinations inferred from feature flags."""
    _register_defaults()
    load_propagation_plugins()

    regimes: dict[str, dict[str, set[str]]] = {}
    for mode, engine in list_registered_propagation_backends():
        features = get_propagation_backend_features(mode, engine)
        derivative_methods = {
            feature.split(".", 1)[1]
            for feature in features
            if feature.startswith("derivative.") and "." in feature
        }
        if not derivative_methods:
            continue

        mode_regimes = regimes.setdefault(mode, {})
        mode_regimes[engine] = derivative_methods

    return regimes


def register_propagation_backend(
    mode: str,
    engine: str,
    factory: PropagationBackendFactory,
    *,
    override: bool = False,
) -> None:
    """Register a propagation backend factory for a mode/engine pair."""
    key = (mode, engine)
    if key in _registry and not override:
        raise ValueError(
            f"Tracer already registered for mode={mode!r}, engine={engine!r}."
        )
    _registry[key] = factory


def get_propagation_backend_factory(
    mode: str, engine: str
) -> Optional[PropagationBackendFactory]:
    """Return the registered propagation backend factory for a mode/engine pair."""
    return _registry.get((mode, engine))


def list_registered_propagation_backends() -> tuple[tuple[str, str], ...]:
    """Return registered backend keys sorted for deterministic introspection."""
    return tuple(sorted(_registry))


def _iter_plugin_entry_points() -> list[_EntryPointLike]:
    """Return entry points for external backend plugins across Python versions."""
    try:
        return cast(
            list[_EntryPointLike], list(metadata.entry_points(group=ENTRY_POINT_GROUP))
        )
    except TypeError:
        all_entry_points = metadata.entry_points()
        if hasattr(all_entry_points, "select"):
            return cast(
                list[_EntryPointLike],
                list(all_entry_points.select(group=ENTRY_POINT_GROUP)),
            )
        legacy_entry_points = cast(Any, all_entry_points)
        return cast(
            list[_EntryPointLike], list(legacy_entry_points.get(ENTRY_POINT_GROUP, []))
        )


def load_propagation_plugins(*, override: bool = False) -> tuple[tuple[str, str], ...]:
    """Load propagation plugins from package entry points and register factories.

    Plugins are expected to declare entry points under
    ``hypy.propagation.backends`` (or the legacy ``hypy.propagation.tracers``)
    with names in the format ``mode:engine`` and values pointing to callables
    compatible with ``PropagationBackendFactory``.
    """
    global _plugins_loaded

    if _plugins_loaded and not override:
        return ()

    loaded: list[tuple[str, str]] = []
    for entry_point in _iter_plugin_entry_points():
        name = str(getattr(entry_point, "name", "")).strip()
        if ":" not in name:
            continue

        mode, engine = (part.strip() for part in name.split(":", 1))
        if not mode or not engine:
            continue

        loaded_object = entry_point.load()
        if not callable(loaded_object):
            continue
        factory = cast(PropagationBackendFactory, loaded_object)

        try:
            register_propagation_backend(mode, engine, factory, override=override)
        except ValueError:
            continue

        loaded.append((mode, engine))

    _plugins_loaded = True
    return tuple(loaded)


def _require_table_grid_params(
    max_depth: Optional[float],
    max_distance: Optional[float],
    depth_step: Optional[float],
    distance_step: Optional[float],
) -> tuple[float, float, float, float]:
    if any(
        value is None for value in (max_depth, max_distance, depth_step, distance_step)
    ):
        raise ValueError(
            "In table mode these variables are required: "
            "max_depth, max_distance, depth_step, distance_step "
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


def _register_defaults() -> None:
    def _table_1d_fmm(
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

        max_depth_f, max_distance_f, depth_step_f, distance_step_f = (
            _require_table_grid_params(
                max_depth=max_depth,
                max_distance=max_distance,
                depth_step=depth_step,
                distance_step=distance_step,
            )
        )
        return FmmTravelTimeGrid1D(
            model_tvel=model_path.as_posix(),
            phases=list(phases),
            max_depth=max_depth_f,
            max_distance=max_distance_f,
            depth_step=depth_step_f,
            distance_step=distance_step_f,
        )

    def _table_1d_tables(
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

        max_depth_f, max_distance_f, depth_step_f, distance_step_f = (
            _require_table_grid_params(
                max_depth=max_depth,
                max_distance=max_distance,
                depth_step=depth_step,
                distance_step=distance_step,
            )
        )
        return TauTravelTimeGrid(
            model_tvel=model_path.as_posix(),
            phases=list(phases),
            max_depth=max_depth_f,
            max_distance=max_distance_f,
            depth_step=depth_step_f,
            distance_step=distance_step_f,
            engine=engine,
        )

    def _ray_1d_taup(*, model_path: Path, **_: object) -> PropagationEngine:
        from hypy.propagation.layered.ray_tracer.taup import TauPRayTracer

        return TauPRayTracer(model_path=model_path.as_posix())

    def _ray_1d_pyrocko(*, model_path: Path, **_: object) -> PropagationEngine:
        from hypy.propagation.layered.ray_tracer.pyrocko import PyrockoRayTracer

        return PyrockoRayTracer(model_path=model_path.as_posix())

    def _table_3d_fmm(
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

        max_depth_f, max_distance_f, depth_step_f, distance_step_f = (
            _require_table_grid_params(
                max_depth=max_depth,
                max_distance=max_distance,
                depth_step=depth_step,
                distance_step=distance_step,
            )
        )
        return FmmTravelTimeGrid3D(
            model_path=model_path.as_posix(),
            phases=list(phases),
            max_depth_km=max_depth_f,
            max_distance_km=max_distance_f,
            depth_step_km=depth_step_f,
            distance_step_km=distance_step_f,
        )

    if get_propagation_backend_factory("table-1d", "fmm") is None:
        register_propagation_backend("table-1d", "fmm", _table_1d_fmm)
    if get_propagation_backend_factory("table-1d", "taup") is None:
        register_propagation_backend("table-1d", "taup", _table_1d_tables)
    if get_propagation_backend_factory("table-1d", "pyrocko") is None:
        register_propagation_backend("table-1d", "pyrocko", _table_1d_tables)
    if get_propagation_backend_factory("ray-1d", "taup") is None:
        register_propagation_backend("ray-1d", "taup", _ray_1d_taup)
    if get_propagation_backend_factory("ray-1d", "pyrocko") is None:
        register_propagation_backend("ray-1d", "pyrocko", _ray_1d_pyrocko)
    if get_propagation_backend_factory("table-3d", "fmm") is None:
        register_propagation_backend("table-3d", "fmm", _table_3d_fmm)


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
    """Create a propagation backend from the mode/engine registry."""
    _register_defaults()
    load_propagation_plugins()
    factory = get_propagation_backend_factory(mode, engine)
    if factory is None:
        if mode not in {registered_mode for registered_mode, _ in _registry}:
            raise ValueError(f"Unsupported mode: {mode}")
        raise ValueError(f"Unsupported engine: {engine} for mode {mode}")

    backend = factory(
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
    backend.feature_flags = BACKEND_FEATURE_FLAGS.get((mode, engine), frozenset())
    backend.required_model_features = REQUIRED_MODEL_FEATURE_FLAGS.get(
        (mode, engine),
        frozenset(),
    )

    if backend.required_model_features:
        model_instance = velocity_model or VelocityModel.create_model(model_path)
        missing = sorted(
            backend.required_model_features.difference(
                model_instance.supported_features
            )
        )
        if missing:
            raise ValueError(
                "Incompatible model and propagation backend: "
                f"mode={mode!r}, engine={engine!r} requires model features "
                f"{missing}, but model {model_instance.model_name!r} "
                f"provides {sorted(model_instance.supported_features)}."
            )

    return backend


TracerFactory = PropagationBackendFactory
register_tracer = register_propagation_backend
get_tracer_factory = get_propagation_backend_factory
list_registered_tracers = list_registered_propagation_backends
load_tracer_plugins = load_propagation_plugins
create_tracer = create_propagation_backend
