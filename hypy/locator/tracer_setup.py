from hypy.propagation import PropagationModel
from hypy.propagation.layered.grid import TauTimeTables

from .types import EngineLiteral, ModeLiteral, Optional, Path


def init_tracer(
    mode: ModeLiteral,
    engine: EngineLiteral,
    phases: tuple[str, ...],
    tvel_file_path: Path,
    max_depth: Optional[float],
    max_distance: Optional[float],
    depth_step: Optional[float],
    distance_step: Optional[float],
) -> PropagationModel:
    if mode == "table":
        if any(
            val is None for val in [max_depth, max_distance, depth_step, distance_step]
        ):
            raise ValueError(
                "In table mode these variables are required: "
                "max_depth, max_distance, depth_step, distance_step "
            )
        return TauTimeTables(
            tvel_file_path.as_posix(),
            list(phases),
            float(max_depth),  # type: ignore
            float(max_distance),  # type: ignore
            float(depth_step),  # type: ignore
            float(distance_step),  # type: ignore
            engine,
        )
    elif mode == "ray":
        if engine == "taup":
            from hypy.propagation.layered.ray_tracer.taup import TauPRayTracer

            return TauPRayTracer(model_path=tvel_file_path.as_posix())
        elif engine == "pyrocko":
            from hypy.propagation.layered.ray_tracer.pyrocko import PyrockoRayTracer

            return PyrockoRayTracer(model_path=tvel_file_path.as_posix())
    else:
        raise ValueError(f"Unsupported mode: {mode}")
