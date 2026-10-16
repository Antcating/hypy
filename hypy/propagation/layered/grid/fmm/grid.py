"""First-arrival travel-time tables using scikit-fmm.

This backend is a fast-eikonal alternative to TauP/Pyrocko-based travel-time
generation. It computes first-arrival travel times in a laterally homogeneous
1D medium sampled on a 2D depth-distance grid.

Notes
-----
- ``"P"`` and ``"S"`` phases are interpreted as using ``vp(z)`` and
    ``vs(z)``, respectively.
- This is not a phase-branch tracer like TauP.
- ``takeoff_angle`` is set to ``0.0`` everywhere by design.
"""

import os
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import skfmm

from hypy.logger import logger
from hypy.model import TvelModel
from hypy.propagation.layered.grid import LayeredGrid1D, model_fingerprint
from hypy.utils.seismic import normalize_phase

_TABLE_CACHE_VERSION = "v2"


class FmmTravelTimeGrid1D(LayeredGrid1D):
    """First-arrival travel-time tables using scikit-fmm.

    where:
        - depth_index corresponds to source depth
        - distance_index corresponds to epicentral distance
    """

    feature_flags = frozenset(
        {
            "propagation.layered",
            "propagation.ray",
            "propagation.table",
            "derivative.numerical",
            "derivative.batch",
        }
    )
    required_model_features = frozenset({"model.profile.1d"})

    def __init__(
        self,
        model: TvelModel,
        phases: Sequence[str],
        max_depth_km: float,
        max_distance_km: float,
        depth_step_km: float,
        distance_step_km: float,
        overwrite: bool = False,
    ):
        normalized_phases = tuple(normalize_phase(phase) for phase in phases)
        super().__init__(
            model,
            mode="table",
            engine="skfmm",
            max_depth_km=max_depth_km,
            max_distance_km=max_distance_km,
            depth_step_km=depth_step_km,
            distance_step_km=distance_step_km,
            phases=normalized_phases,
        )

        self.model = model
        self.model_name = model.model_name

        # Backward-compatible aliases.  The locator uses the canonical *_km
        # attributes initialized by PropagationEngine.  Previously only these
        # aliases were populated, leaving max_depth_km at the planet radius and
        # max_distance_km/depth_step_km/distance_step_km unset.
        self.max_depth = self.max_depth_km
        self.max_distance = self.max_distance_km
        self.depth_step = self.depth_step_km
        self.distance_step = self.distance_step_km

        self.depth_grid: np.ndarray
        self.distance_grid: np.ndarray

        self.travel_time_tables: Dict[str, np.ndarray] = {}
        self.takeoff_angle_tables: Dict[str, np.ndarray] = {}

        self._build_tables(overwrite=overwrite)

    def _table_filename(self, phase: str) -> str:
        table_root_dir = Path(__file__).parent
        model_hash = model_fingerprint(self.velocity_model)
        table_name = (
            f"travel_time_tables_{self.model_name}_{model_hash}_{_TABLE_CACHE_VERSION}"
            f"_{self.engine}_{phase}_d{self.max_depth}_r{self.max_distance}"
            f"_dz{self.depth_step}_dr{self.distance_step}.npz"
        )
        return os.path.join(table_root_dir, "data", table_name)

    def _build_velocity_profiles(self) -> tuple[np.ndarray, np.ndarray]:
        """Build ``vp(z)`` and ``vs(z)`` sampled on ``self.depth_grid``."""
        depths = self.depth_grid

        vp_profile = np.full(depths.shape, np.nan, dtype=float)
        vs_profile = np.full(depths.shape, np.nan, dtype=float)

        # layer = [depth_start, depth_end, vp_start, vp_end, vs_start, vs_end, ...]
        for layer in self.model.model.model.s_mod.v_mod.layers:
            z_start = float(layer[0])
            z_end = float(layer[1])

            vp_start = float(layer[2])
            vp_end = float(layer[3])

            vs_start = float(layer[4])
            vs_end = float(layer[5])

            z0 = max(z_start, float(depths[0]))
            z1 = min(z_end, float(depths[-1]))

            mask = (depths >= z0) & (depths <= z1)
            if not np.any(mask):
                continue

            z = depths[mask]
            dz = z_end - z_start

            if abs(dz) < 1e-12:
                frac = np.zeros_like(z)
            else:
                frac = (z - z_start) / dz

            vp_profile[mask] = vp_start + frac * (vp_end - vp_start)
            vs_profile[mask] = vs_start + frac * (vs_end - vs_start)

        if np.any(np.isnan(vp_profile)):
            raise ValueError("vp_profile contains NaNs. Check model depth coverage.")
        if np.any(np.isnan(vs_profile)):
            raise ValueError("vs_profile contains NaNs. Check model depth coverage.")

        return vp_profile, vs_profile

    def _solve_fmm_surface_times(
        self,
        velocity_grid: np.ndarray,
        source_depth_index: int,
    ) -> np.ndarray:
        """Solve eikonal travel times for one source depth.

        Returns surface travel times as a function of horizontal
        distance. Axis 0 is depth and axis 1 is horizontal distance.
        """
        phi = np.ones_like(velocity_grid, dtype=float)

        # Source at distance = 0, depth = source_depth_index
        phi[source_depth_index, 0] = -1.0

        if self.depth_step != self.distance_step:
            raise NotImplementedError(
                "Anisotropic grid spacing is not implemented in this method."
            )
        tt = skfmm.travel_time(
            phi,
            speed=velocity_grid,
            dx=self.depth_step,
        )

        # Receivers are assumed at surface z=0
        surface_tt = tt[0, :]

        return np.asarray(surface_tt, dtype=float)

    def _build_tables(self, overwrite: bool = False) -> None:
        """Build or load first-arrival travel-time tables for each phase."""
        self.depth_grid = np.arange(
            0.0, self.max_depth + 0.0 + 1e-6, self.depth_step, dtype=float
        )
        self.distance_grid = np.arange(
            0.0, self.max_distance + 0.0 + 1e-6, self.distance_step, dtype=float
        )

        vp_profile, vs_profile = self._build_velocity_profiles()

        vp_grid = np.broadcast_to(
            vp_profile[:, None],
            (self.depth_grid.size, self.distance_grid.size),
        ).copy()
        vs_grid = np.broadcast_to(
            vs_profile[:, None],
            (self.depth_grid.size, self.distance_grid.size),
        ).copy()

        for phase in self.phases:
            fname = self._table_filename(phase)

            if os.path.exists(fname) and not overwrite:
                logger.info(f"Loading travel-time tables from {fname}")
                self.load(fname, phase)
                continue

            logger.info(f"Building scikit-fmm travel-time table for phase {phase}")

            tt_table = np.full(
                (self.depth_grid.size, self.distance_grid.size),
                np.nan,
                dtype=float,
            )
            ta_table = np.zeros_like(tt_table, dtype=float)

            if phase.upper() == "P":
                velocity_grid = vp_grid
            elif phase.upper() == "S":
                velocity_grid = vs_grid
            else:
                raise ValueError(
                    f"Unsupported phase '{phase}' for scikit-fmm tables. "
                    "Use only 'P' and/or 'S'."
                )

            for i in range(self.depth_grid.size):
                tt_surface = self._solve_fmm_surface_times(
                    velocity_grid=velocity_grid,
                    source_depth_index=i,
                )
                tt_table[i, :] = tt_surface

                # Requested behavior:
                # takeoff angle should be zero
                ta_table[i, :] = 0.0

            self.travel_time_tables[phase] = tt_table
            self.takeoff_angle_tables[phase] = ta_table

            logger.info(f"Saving travel-time tables to {fname}")
            self.save(fname)
