from ..propagation import PropagationModel
from ..propagation.ray_tracers import RayTracer
from ..propagation.grid.tau import TauTimeTables
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
        return RayTracer(model_path=tvel_file_path.as_posix(), engine=engine)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
