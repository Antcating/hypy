from typing import Dict, Tuple

import numpy as np
from obspy.taup.taup_create import build_taup_model  # type: ignore[import-untyped]

from .types import TauPyModel, Path


def _init_velocity_model(model_name: str, tvel_file_path: Path) -> TauPyModel:
    try:
        return TauPyModel(model_name)
    except FileNotFoundError:
        # Build TauP model from .tvel if missing, then load
        build_taup_model(tvel_file_path.as_posix())
        return TauPyModel(model_name)


def precompute_velocity_profiles_1d(
    *,
    model: TauPyModel,
    phases: Tuple[str, ...],
    max_depth: float,
    depth_step: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Precompute velocity profiles for each phase.

    Returns a dict: phase_key -> (depths_km, velocities_km_s).
    """
    velocity_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    depths = np.arange(0, max_depth + depth_step, depth_step)
    for phase in phases:
        phase_lower = phase.lower()[-1]
        v_profile = []
        if model.model.s_mod is None:
            raise ValueError("TauP velocity model 's_mod' is not initialized.")
        for d in depths:
            v = model.model.s_mod.v_mod.evaluate_below(d, phase_lower)[0]
            v_profile.append(v)
        velocity_profiles[phase_lower] = (depths, np.array(v_profile))

    return velocity_profiles


def get_velocity_1d(
    *,
    model: TauPyModel,
    precalculated: bool,
    velocity_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]],
    depth_km: float,
    phase: str,
) -> float:
    """Get velocity at a given depth for a specific phase."""
    phase_lower = phase.lower()[-1]
    if not precalculated:
        return float(model.model.s_mod.v_mod.evaluate_below(depth_km, phase_lower)[0])
    depths, v_profile = velocity_profiles[phase_lower]
    return float(np.interp(depth_km, depths, v_profile))
