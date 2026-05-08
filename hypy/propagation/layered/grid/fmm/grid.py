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
- ``ray_param`` is set to ``0.0`` everywhere by design.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import skfmm

from hypy.logger import logger  # type: ignore[import-not-found]
from hypy.propagation.layered import LayeredPropagationEngine
from hypy.utils.geo import gps2dist_azimuth_nb
from hypy.utils.math import interp_quant


class FmmTravelTimeGrid1D(LayeredPropagationEngine):
    """
    First-arrival travel-time tables using scikit-fmm.

    where:
        - depth_index corresponds to source depth
        - distance_index corresponds to epicentral distance
    """

    def __init__(
        self,
        model_tvel: str,
        phases: List[str],
        max_depth: float,
        max_distance: float,
        depth_step: float,
        distance_step: float,
        overwrite: bool = False,
    ):
        super().__init__(model_tvel)

        self.engine = "skfmm"
        self.model_name = Path(model_tvel).stem
        self.phases = phases
        self.max_depth = max_depth
        self.max_distance = max_distance
        self.depth_step = depth_step
        self.distance_step = distance_step

        self.depth_grid: np.ndarray
        self.distance_grid: np.ndarray

        self.travel_time_tables: Dict[str, np.ndarray] = {}
        self.ray_param_tables: Dict[str, np.ndarray] = {}
        self.takeoff_angle_tables: Dict[str, np.ndarray] = {}

        self._build_tables(overwrite=overwrite)

    def _table_filename(self, phase: str) -> str:
        table_root_dir = Path(__file__).parent
        table_name = (
            f"travel_time_tables_{self.model_name}_{self.engine}_{phase}"
            f"_d{self.max_depth}_r{self.max_distance}"
            f"_dz{self.depth_step}_dr{self.distance_step}.npz"
        )
        return os.path.join(table_root_dir, "data", table_name)

    def save(self, filename: str) -> None:
        """Save the travel-time, ray-parameter, and takeoff-angle tables."""
        if self.depth_grid is None or self.distance_grid is None:
            raise ValueError("Travel-time tables are not built; nothing to save.")

        save_dict = {
            "depth_grid": self.depth_grid,
            "distance_grid": self.distance_grid,
        }

        for ph in self.travel_time_tables:
            save_dict[f"tt_{ph}"] = self.travel_time_tables[ph]
        for ph in self.ray_param_tables:
            save_dict[f"rp_{ph}"] = self.ray_param_tables[ph]
        for ph in self.takeoff_angle_tables:
            save_dict[f"ta_{ph}"] = self.takeoff_angle_tables[ph]

        np.savez_compressed(filename, allow_pickle=True, **save_dict)

    def load(self, filename: str, phase: str) -> None:
        """Load travel-time, ray-parameter, and takeoff-angle tables."""
        data = np.load(filename, allow_pickle=True)
        self.depth_grid = data["depth_grid"]
        self.distance_grid = data["distance_grid"]

        if f"tt_{phase}" in data:
            self.travel_time_tables[phase] = data[f"tt_{phase}"]
        if f"rp_{phase}" in data:
            self.ray_param_tables[phase] = data[f"rp_{phase}"]
        if f"ta_{phase}" in data:
            self.takeoff_angle_tables[phase] = data[f"ta_{phase}"]

    def _build_velocity_profiles(self) -> tuple[np.ndarray, np.ndarray]:
        """Build ``vp(z)`` and ``vs(z)`` sampled on ``self.depth_grid``."""
        depths = self.depth_grid

        vp_profile = np.full(depths.shape, np.nan, dtype=float)
        vs_profile = np.full(depths.shape, np.nan, dtype=float)

        # Same layer structure you used in your prototype:
        # layer = [depth_start, depth_end, vp_start, vp_end, vs_start, vs_end, ...]
        for layer in self.velocity_model.model.model.s_mod.v_mod.layers:
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

        Returns surface travel times as a function of horizontal distance.
        Axis 0 is depth and axis 1 is horizontal distance.
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
            rp_table = np.zeros_like(tt_table, dtype=float)
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
                rp_table[i, :] = 0.0

            self.travel_time_tables[phase] = tt_table
            self.ray_param_tables[phase] = rp_table
            self.takeoff_angle_tables[phase] = ta_table

            logger.info(f"Saving travel-time tables to {fname}")
            Path(fname).parent.mkdir(parents=True, exist_ok=True)
            self.save(fname)

    def compute_travel_time(self, phase: str, depth: float, distance: float) -> float:
        """Interpolate the first-arrival travel time for a phase."""
        return interp_quant(
            depth,
            distance,
            self.depth_grid,
            self.distance_grid,
            self.travel_time_tables[phase],
        )

    def compute_takeoff_angle(self, phase: str, depth: float, distance: float) -> float:
        """Interpolate the takeoff angle for a phase."""
        return interp_quant(
            depth,
            distance,
            self.depth_grid,
            self.distance_grid,
            self.takeoff_angle_tables[phase],
        )

    def get_arrival(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_elevation_m: float,
        phase: str,
        max_distance_km: float | None = None,
    ) -> tuple[float, Optional[float], Optional[float], float]:
        """Return travel-time table values for a source-receiver pair.

        Returns
        -------
        tuple of float or None
            ``(arc_distance_km, travel_time_s, takeoff_angle_deg, azimuth_deg)``.
        """
        if phase not in self.phases:
            raise ValueError(f"Phase {phase} not in available phases: {self.phases}")

        dist_m, azimuth_deg, _ = gps2dist_azimuth_nb(
            source_latitude,
            source_longitude,
            receiver_latitude,
            receiver_longitude,
            R=self.velocity_model.model.model.radius_of_planet * 1000.0,
        )
        arc_distance_km = dist_m / 1000.0

        if max_distance_km is not None and arc_distance_km > max_distance_km:
            logger.info(
                "Station at distance %.2f km exceeds max_distance %.2f km. Skipping.",
                arc_distance_km,
                max_distance_km,
            )
            return arc_distance_km, None, None, azimuth_deg

        source_depth_eff_km = source_depth_km + receiver_elevation_m / 1000.0

        time = self.compute_travel_time(phase, source_depth_eff_km, arc_distance_km)
        takeoff_angle = self.compute_takeoff_angle(
            phase, source_depth_eff_km, arc_distance_km
        )

        if np.isnan(time) or np.isnan(takeoff_angle):
            raise ValueError(
                f"No travel time or takeoff angle found for phase {phase} "
                f"at depth {source_depth_eff_km} and distance {arc_distance_km}."
            )

        return arc_distance_km, time, takeoff_angle, azimuth_deg

    def get_arrival_1d(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ) -> tuple[float, float]:
        """Return travel time and takeoff angle for a 1D query.

        Returns
        -------
        tuple of float
            ``(travel_time_s, takeoff_angle_deg)``.
        """
        if phase not in self.phases:
            raise ValueError(f"Phase {phase} not in available phases: {self.phases}")

        time = self.compute_travel_time(phase, source_depth_km, arc_distance_km)
        takeoff_angle = self.compute_takeoff_angle(
            phase, source_depth_km, arc_distance_km
        )

        if np.isnan(time) or np.isnan(takeoff_angle):
            raise ValueError(
                f"No travel time or takeoff angle found for phase {phase} "
                f"at depth {source_depth_km} and distance {arc_distance_km}."
            )

        return (
            time,
            None,
        )  # takeoff angle is not meaningful for scikit-fmm tables, set to None

    def compute_derivatives(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        az_deg: float,
        phase: str,
        takeoff_angle_deg: Optional[float],
        delta: Optional[float] = 0.01,
        mode: str = "analytical",
    ):
        """Compute spatial slowness derivatives with finite differences.

        Returns
        -------
        tuple of float
            ``(dt_dr, dt_dx, dt_dy, dt_dz)`` in s/km.
        """
        if mode == "analytical":
            raise NotImplementedError(
                "Analytical derivatives are not implemented for scikit-fmm tables."
            )
        elif mode == "numerical":
            if delta is None:
                raise ValueError(
                    "Delta must be provided for numerical derivative computation."
                )
            # TODO: Expose numerical derivative stencils (central, forward,
            # backward, five-point, etc.).
            return self._compute_derivatives_numerical(
                arc_distance_km=arc_distance_km,
                source_depth_km=source_depth_km,
                azimuth_deg=az_deg,
                phase=phase,
                delta=delta,
            )
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Supported modes are 'analytical' and "
                "'numerical'."
            )
