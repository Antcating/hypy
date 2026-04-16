"""Support for layered velocity models stored in ``.tvel`` files."""

from pathlib import Path
from typing import Sequence

import numpy as np
from obspy.taup import TauPyModel  # type: ignore[import-untyped]
from obspy.taup.taup_create import build_taup_model  # type: ignore[import-untyped]

from hypy.model import VelocityModel
from hypy.utils.math import velocity_from_layers


class TvelModel(VelocityModel):
    """1D velocity model driver for .tvel files via ObsPy TauPyModel."""

    feature_flags = frozenset(
        {
            "model.layered",
            "model.profile.1d",
            "format.tvel",
            "phase.p",
            "phase.s",
        }
    )

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
        self._layer_top_depth: np.ndarray | None = None
        self._layer_bot_depth: np.ndarray | None = None
        self._layer_top_p_velocity: np.ndarray | None = None
        self._layer_bot_p_velocity: np.ndarray | None = None
        self._layer_top_s_velocity: np.ndarray | None = None
        self._layer_bot_s_velocity: np.ndarray | None = None

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

        cache_dir = self.model_path.parent / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for phase in phases:
            phase_lower = phase.lower()[-1]
            cache_file = cache_dir / (
                f"tvel_profile_{self.model_name}_{phase_lower}_"
                f"{max_depth}_{depth_step}.npz"
            )
            if cache_file.exists():
                with np.load(cache_file, allow_pickle=True) as loaded:
                    self.velocity_profiles[phase_lower] = (
                        loaded["depths"],
                        loaded["v_profile"],
                    )
                continue

            v_profile = self._build_velocity_profile(phase_lower, depths)
            self.velocity_profiles[phase_lower] = (depths, v_profile)
            np.savez_compressed(
                cache_file,
                max_depth=max_depth,
                depth_step=depth_step,
                phase=phase_lower,
                depths=depths,
                v_profile=v_profile,
            )

    def _build_velocity_profile(
        self,
        phase_lower: str,
        depths: np.ndarray,
    ) -> np.ndarray:
        if self.model.model.s_mod is None:
            raise ValueError("TauP velocity model 's_mod' is not initialized.")
        return np.array(
            [
                self.model.model.s_mod.v_mod.evaluate_below(depth, phase_lower)[0]
                for depth in depths
            ]
        )

    def _initialize_layer_arrays(self) -> None:
        if self._layer_top_depth is not None:
            return

        if self.model.model.s_mod is None:
            raise ValueError("TauP velocity model 's_mod' is not initialized.")

        layers = self.model.model.s_mod.v_mod.layers
        if getattr(layers.dtype, "names", None) is not None:
            self._layer_top_depth = np.asarray(layers["top_depth"], dtype=float)
            self._layer_bot_depth = np.asarray(layers["bot_depth"], dtype=float)
            self._layer_top_p_velocity = np.asarray(
                layers["top_p_velocity"], dtype=float
            )
            self._layer_bot_p_velocity = np.asarray(
                layers["bot_p_velocity"], dtype=float
            )
            self._layer_top_s_velocity = np.asarray(
                layers["top_s_velocity"], dtype=float
            )
            self._layer_bot_s_velocity = np.asarray(
                layers["bot_s_velocity"], dtype=float
            )
        else:
            self._layer_top_depth = np.asarray(layers[:, 0], dtype=float)
            self._layer_bot_depth = np.asarray(layers[:, 1], dtype=float)
            self._layer_top_p_velocity = np.asarray(layers[:, 2], dtype=float)
            self._layer_bot_p_velocity = np.asarray(layers[:, 3], dtype=float)
            self._layer_top_s_velocity = np.asarray(layers[:, 4], dtype=float)
            self._layer_bot_s_velocity = np.asarray(layers[:, 5], dtype=float)

    def _velocity_from_layers(self, depth: float, phase_lower: str) -> float:
        self._initialize_layer_arrays()
        assert self._layer_top_depth is not None
        assert self._layer_bot_depth is not None
        assert self._layer_top_p_velocity is not None
        assert self._layer_bot_p_velocity is not None
        assert self._layer_top_s_velocity is not None
        assert self._layer_bot_s_velocity is not None

        top_depth = self._layer_top_depth
        bot_depth = self._layer_bot_depth
        if phase_lower == "p":
            top_velocity = self._layer_top_p_velocity
            bot_velocity = self._layer_bot_p_velocity
        else:
            top_velocity = self._layer_top_s_velocity
            bot_velocity = self._layer_bot_s_velocity

        if depth < top_depth[0] or depth > bot_depth[-1]:
            raise ValueError(
                f"Depth {depth} km is outside model bounds "
                f"[{top_depth[0]}, {bot_depth[-1]}]."
            )

        return float(
            velocity_from_layers(
                depth,
                top_depth,
                bot_depth,
                top_velocity,
                bot_velocity,
            )
        )

    def get_velocity_1d(self, depth: float, phase: str) -> float:
        """Return velocity (km/s) at depth for signed phase."""
        phase_lower = phase.lower()[-1]

        if self.velocity_profiles and phase_lower in self.velocity_profiles:
            depths, v_profile = self.velocity_profiles[phase_lower]
            return float(np.interp(depth, depths, v_profile))

        return self._velocity_from_layers(depth, phase_lower)
