import numpy as np
from obspy.taup.taup_create import build_taup_model  # type: ignore[import-untyped]

from .types import TauPyModel, Path


class VelocityModelHandler:
    def __init__(self, tvel_file_path: Path):
        self.model = self._init_velocity_model(tvel_file_path.stem, tvel_file_path)
        self.velocity_profiles: dict = {}

    @staticmethod
    def _init_velocity_model(model_name: str, tvel_file_path: Path):
        try:
            return TauPyModel(model_name)
        except FileNotFoundError:
            # Build TauP model from .tvel if missing, then load
            build_taup_model(tvel_file_path.as_posix())
            return TauPyModel(model_name)

    def _precompute_velocity_profiles_1d(self, phases, max_depth, depth_step):
        """Precompute and store velocity as a function of depth for each phase."""
        # Reinitializing the velocity_profiles dictionary to store the precomputed profiles
        self.velocity_profiles = {}
        depths = np.arange(0, max_depth + depth_step, depth_step)
        for phase in phases:
            phase_lower = phase.lower()[-1]
            v_profile = []
            if self.model.model.s_mod is None:
                raise ValueError("TauP velocity model 's_mod' is not initialized.")
            for d in depths:
                v = self.model.model.s_mod.v_mod.evaluate_below(d, phase_lower)[0]
                v_profile.append(v)
            self.velocity_profiles[phase_lower] = (depths, np.array(v_profile))

    # TODO: Write a comparison function to evaluate the accuracy of
    # precomputed velocities against direct TauP calls,
    # especially for ray-tracing mode where accuracy is critical.
    def _get_velocity_1d(self, depth, phase):
        """Get velocity at a given depth for a specific phase."""
        phase_lower = phase.lower()[-1]
        if self.velocity_profiles and phase_lower in self.velocity_profiles:
            depths, v_profile = self.velocity_profiles[phase_lower]
            # TODO: Find a more accurate interpolation method that can
            # handle the step function nature of velocity models,
            # especially at lower sampling rates.
            return float(np.interp(depth, depths, v_profile))
        return self.model.model.s_mod.v_mod.evaluate_below(depth, phase_lower)[0]
