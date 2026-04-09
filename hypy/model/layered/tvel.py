"""Support for layered velocity models stored in ``.tvel`` files."""

from pathlib import Path
from typing import Sequence

import numpy as np
from obspy.taup import TauPyModel  # type: ignore[import-untyped]
from obspy.taup.taup_create import build_taup_model  # type: ignore[import-untyped]

from hypy.model import VelocityModel


class TvelModel(VelocityModel):
    """1D velocity model driver for .tvel files via ObsPy TauPyModel."""

    def __init__(self, model_path: Path | str, dim: int | str = "auto"):
        model_path = Path(model_path).expanduser().resolve()

        super().__init__(model_path=model_path, model_name=model_path.stem, dim=1)

        if not model_path.exists():
            raise FileNotFoundError(f"Velocity model file not found: {model_path}")

        if model_path.suffix.lower() != ".tvel":
            raise ValueError(
                f"Unsupported velocity model file type: {model_path.suffix}. "
                "Only .tvel files are supported."
            )

        self.model_name = model_path.stem
        if dim != "auto" and dim != 1:
            raise ValueError(
                f"Unsupported model dimension '{dim}' for .tvel file. "
                "Only 1D models are supported."
            )

        self.dim = 1
        self.model = self._init_velocity_model(self.model_name, model_path)
        self.velocity_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _init_velocity_model(model_name: str, tvel_file_path: Path) -> TauPyModel:
        try:
            return TauPyModel(model_name)
        except FileNotFoundError:
            build_taup_model(tvel_file_path.as_posix())
            return TauPyModel(model_name)

    def _precompute_velocity_profiles_1d(
        self,
        phases: Sequence[str],
        max_depth: float,
        depth_step: float,
    ) -> None:
        """Precompute and store velocity profiles by phase."""
        self.velocity_profiles = {}
        depths = np.arange(0, max_depth + depth_step, depth_step)

        if self.model.model.s_mod is None:
            raise ValueError("TauP velocity model 's_mod' is not initialized.")

        for phase in phases:
            phase_lower = phase.lower()[-1]
            v_profile = []
            for d in depths:
                v = self.model.model.s_mod.v_mod.evaluate_below(d, phase_lower)[0]
                v_profile.append(v)
            self.velocity_profiles[phase_lower] = (depths, np.array(v_profile))

    # Backward compatibility alias
    precompute_velocity_profiles_1d = _precompute_velocity_profiles_1d

    def get_velocity_1d(self, depth: float, phase: str) -> float:
        """Return velocity (km/s) at depth for signed phase."""
        phase_lower = phase.lower()[-1]
        if self.velocity_profiles and phase_lower in self.velocity_profiles:
            depths, v_profile = self.velocity_profiles[phase_lower]
            return float(np.interp(depth, depths, v_profile))

        if self.model.model.s_mod is None:
            raise ValueError("TauP velocity model 's_mod' is not initialized.")

        return float(self.model.model.s_mod.v_mod.evaluate_below(depth, phase_lower)[0])


# Backward compatibility class name used by locator modules.
VelocityModelHandler = TvelModel
