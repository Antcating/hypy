from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
from .types import Optional, np

class ModelMixin:
    model: TauPyModel
    _precalculate_velocities: bool
    phases: tuple[str, ...]
    velocity_profiles: dict

    @staticmethod
    def _init_velocity_model(model_name: str, tvel_path):
        try:
            return TauPyModel(model_name)
        except FileNotFoundError:
            # Build TauP model from .tvel if missing, then load
            build_taup_model(tvel_path.as_posix())
            return TauPyModel(model_name)

    def _precompute_velocity_profiles_1d(self, max_depth, depth_step):
        """Precompute and store velocity as a function of depth for each phase."""
        self.velocity_profiles = {}
        depths = np.arange(0, max_depth + depth_step, depth_step)
        for phase in self.phases:
            phase_lower = phase.lower()
            v_profile = []
            if self.model.model.s_mod is None:
                raise ValueError("TauP velocity model 's_mod' is not initialized.")
            for d in depths:
                v = self.model.model.s_mod.v_mod.evaluate_below(d, phase_lower)[0]
                v_profile.append(v)
            self.velocity_profiles[phase_lower] = (depths, np.array(v_profile))

    def _get_velocity_1d(self, depth, phase):
        """Get velocity at a given depth for a specific phase."""
        phase_lower = phase.lower()
        if not self._precalculate_velocities:
            return self.model.model.s_mod.v_mod.evaluate_below(depth, phase)[0]
        depths, v_profile = self.velocity_profiles[phase_lower]
        return float(np.interp(depth, depths, v_profile))
