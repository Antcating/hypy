from hypy.propagation import PropagationModel
from hypy.propagation.layered.grid import SKFMTimeTables, TauTimeTables

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
    # TODO: Rewrite this to be more flexible and extensible,
    # e.g. by using a registry of available tracers and their required parameters.
    if mode == "table-1d":
        if any(
            val is None for val in [max_depth, max_distance, depth_step, distance_step]
        ):
            raise ValueError(
                "In table mode these variables are required: "
                "max_depth, max_distance, depth_step, distance_step "
            )
        if engine == "fmm":
            return SKFMTimeTables(
                tvel_file_path.as_posix(),
                list(phases),
                max_depth=max_depth,  # type: ignore
                max_distance=max_distance,  # type: ignore
                depth_step=depth_step,  # type: ignore
                distance_step=distance_step,  # type: ignore
            )
        else:
            return TauTimeTables(
                model_tvel=tvel_file_path.as_posix(),
                phases=list(phases),
                max_depth=float(max_depth),  # type: ignore
                max_distance=float(max_distance),  # type: ignore
                depth_step=float(depth_step),  # type: ignore
                distance_step=float(distance_step),  # type: ignore
                engine=engine,
            )
    elif mode == "ray-1d":
        if engine == "taup":
            from hypy.propagation.layered.ray_tracer.taup import TauPRayTracer

            return TauPRayTracer(model_path=tvel_file_path.as_posix())
        elif engine == "pyrocko":
            from hypy.propagation.layered.ray_tracer.pyrocko import PyrockoRayTracer

            return PyrockoRayTracer(model_path=tvel_file_path.as_posix())
    elif mode == "table3d":
        if engine == "fmm":
            from hypy.propagation.volumetric.grid.pyekfmm import PyekfmmRayTracer

            return PyekfmmRayTracer(
                model_path=tvel_file_path.as_posix(),
                max_distance_km=float(max_distance)
                if max_distance is not None
                else None,
                distance_step_km=float(distance_step)
                if distance_step is not None
                else None,
                max_depth_km=float(max_depth) if max_depth is not None else None,
                depth_step_km=float(depth_step) if depth_step is not None else None,
            )
        else:
            raise ValueError(f"Unsupported engine: {engine} for mode {mode}")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return None
