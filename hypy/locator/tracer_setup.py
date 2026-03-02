from ..propagation import PropagationModel
from ..propagation.ray_tracers import RayTracer
from ..propagation.grid.isotropic import TauTimeTables
from ..propagation.grid.anisotropic.ttcrpy.trace3d import Trace3D
from .types import EngineLiteral, ModeLiteral, Optional


def _init_tracer(
    *,
    mode: ModeLiteral,
    engine: EngineLiteral,
    phases: "tuple[str, ...]",
    tvel_file_path,
    model_name: str,
    max_depth: Optional[float],
    max_distance: Optional[float],
    depth_step: Optional[float],
    distance_step: Optional[float],
) -> "tuple[PropagationModel, Optional[float]]":
    tracer: PropagationModel
    if mode == "table":
        missing = [
            name
            for name, val in dict(
                max_depth=max_depth,
                max_distance=max_distance,
                depth_step=depth_step,
                distance_step=distance_step,
            ).items()
            if val is None
        ]
        if missing:
            raise ValueError(
                "In table mode these variables are required: "
                "max_depth, max_distance, depth_step, distance_step "
                f"(missing: {', '.join(missing)})"
            )
        tracer = TauTimeTables(
            tvel_file_path.as_posix(),
            model_name,
            list(phases),
            float(max_depth),
            float(max_distance),
            float(depth_step),
            float(distance_step),
            engine,
        )
        return tracer, float(max_depth)
    elif mode == "3d":
        missing = [
            name
            for name, val in dict(
                max_depth=max_depth,
                max_distance=max_distance,
                depth_step=depth_step,
                distance_step=distance_step,
            ).items()
            if val is None
        ]
        if missing:
            raise ValueError(
                "In 3d mode these variables are required: "
                "max_depth, max_distance, depth_step, distance_step "
                f"(missing: {', '.join(missing)})"
            )
        tracer = Trace3D(
            model_tvel=tvel_file_path.as_posix(),
            model_name=model_name,
            phases=list(phases),
            max_depth=float(max_depth),
            max_distance=float(max_distance),
            depth_step=float(depth_step),
            distance_step=float(distance_step),
            model_center=(10.0, 10.0),
        )
        return tracer, float(max_depth)
    else:
        tracer = RayTracer(model_path=tvel_file_path.as_posix(), engine=engine)
        return tracer, max_depth  # max_depth unused in ray mode
