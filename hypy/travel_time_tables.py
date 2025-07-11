"""
travel_time_tables.py
--------------------
Handles travel-time, ray-parameter, and takeoff angle table creation, saving, loading, and interpolation.
"""

from typing import Dict, List, Optional, Callable
import numpy as np
from obspy.taup import TauPyModel
from obspy.geodetics import kilometer2degrees
import os
from pathlib import Path

from hypy.utils.interp import adaptive_interp_1d


class TravelTimeTables:
    """
    Manages travel-time, ray-parameter, and takeoff angle tables for seismic phases.
    """

    def __init__(
        self,
        model: str,
        phases: List[str],
        max_depth: float,
        max_distance: float,
        depth_step: float,
        distance_step: float,
        overwrite: bool = False,
    ):
        self.model = TauPyModel(model)
        self.model_name = model
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

    def _table_filename(self) -> str:
        phase_str = "_".join(sorted(self.phases))
        table_root_dir = Path(__file__).parent
        table_name = f"travel_time_tables_{self.model_name}_{phase_str}_d{self.max_depth}_r{self.max_distance}_dz{self.depth_step}_dr{self.distance_step}.npz"
        return os.path.join(table_root_dir, "data", table_name)

    def save(self, filename: Optional[str] = None) -> None:
        """
        Save the travel-time, ray-parameter, and takeoff angle tables to a compressed NumPy file.
        """
        if self.depth_grid is None or self.distance_grid is None:
            raise ValueError("Travel-time tables are not built; nothing to save.")
        save_dict = {"depth_grid": self.depth_grid, "distance_grid": self.distance_grid}
        for ph in self.travel_time_tables:
            save_dict[f"tt_{ph}"] = self.travel_time_tables[ph]
        for ph in self.ray_param_tables:
            save_dict[f"rp_{ph}"] = self.ray_param_tables[ph]
        for ph in self.takeoff_angle_tables:
            save_dict[f"ta_{ph}"] = self.takeoff_angle_tables[ph]
        np.savez_compressed(filename or self._table_filename(), **save_dict)

    def load(self, filename: Optional[str] = None) -> None:
        """
        Load travel-time, ray-parameter, and takeoff angle tables from a file.
        """
        data = np.load(filename or self._table_filename(), allow_pickle=True)
        self.depth_grid = data["depth_grid"]
        self.distance_grid = data["distance_grid"]
        self.travel_time_tables = {}
        self.ray_param_tables = {}
        self.takeoff_angle_tables = {}
        for ph in self.phases:
            if f"tt_{ph}" in data:
                self.travel_time_tables[ph] = data[f"tt_{ph}"]
            if f"rp_{ph}" in data:
                self.ray_param_tables[ph] = data[f"rp_{ph}"]
            if f"ta_{ph}" in data:
                self.takeoff_angle_tables[ph] = data[f"ta_{ph}"]

    def _build_tables(self, overwrite: bool = False) -> None:
        """
        Build (or load) the travel-time, ray-parameter, and takeoff angle tables for the given phases.
        """
        fname = self._table_filename()
        if os.path.exists(fname) and not overwrite:
            print(f"Loading travel-time tables from {fname}")
            self.load(fname)
            return
        depth_values = np.arange(0.0, self.max_depth + 1e-6, self.depth_step)
        distance_values = np.arange(0.0, self.max_distance + 1e-6, self.distance_step)
        self.depth_grid = depth_values
        self.distance_grid = distance_values
        for phase in self.phases:
            self.travel_time_tables[phase] = np.full(
                (len(depth_values), len(distance_values)), np.nan
            )
            self.ray_param_tables[phase] = np.full(
                (len(depth_values), len(distance_values)), np.nan
            )
            self.takeoff_angle_tables[phase] = np.full(
                (len(depth_values), len(distance_values)), np.nan
            )
        for i, depth in enumerate(depth_values):
            for j, dist in enumerate(distance_values):
                dist_deg = kilometer2degrees(dist)
                # Handle P phase
                if "P" in self.phases and "P" in self.travel_time_tables:
                    try:
                        arrivals = self.model.get_travel_times(
                            source_depth_in_km=depth,
                            distance_in_degree=dist_deg,
                            phase_list=["ttp"],
                        )
                    except Exception:
                        arrivals = []
                    if arrivals:
                        arr = arrivals[0]
                        if hasattr(arr, "time"):
                            self.travel_time_tables["P"][i, j] = getattr(
                                arr, "time", np.nan
                            )
                        if hasattr(arr, "ray_param"):
                            self.ray_param_tables["P"][i, j] = getattr(
                                arr, "ray_param", np.nan
                            )
                        if hasattr(arr, "takeoff_angle"):
                            self.takeoff_angle_tables["P"][i, j] = getattr(
                                arr, "takeoff_angle", np.nan
                            )
                # Handle S phase
                if "S" in self.phases and "S" in self.travel_time_tables:
                    try:
                        arrivals = self.model.get_travel_times(
                            source_depth_in_km=depth,
                            distance_in_degree=dist_deg,
                            phase_list=["tts"],
                        )
                    except Exception:
                        arrivals = []
                    if arrivals:
                        arr = arrivals[0]
                        if hasattr(arr, "time"):
                            self.travel_time_tables["S"][i, j] = getattr(
                                arr, "time", np.nan
                            )
                        if hasattr(arr, "ray_param"):
                            self.ray_param_tables["S"][i, j] = getattr(
                                arr, "ray_param", np.nan
                            )
                        if hasattr(arr, "takeoff_angle"):
                            self.takeoff_angle_tables["S"][i, j] = getattr(
                                arr, "takeoff_angle", np.nan
                            )
        print(f"Saving travel-time tables to {fname}")
        self.save(fname)

    def interpolate_2d_slice(
        self,
        phase: str,
        depth: float,
        distance: float,
        interp_1d_func: Callable,
    ) -> float:
        """
        Interpolate in (depth, distance) space using adaptive 1D interpolation along distance and linear in depth.

        Parameters:
        - depth, distance: Coordinates to interpolate at
        - depth_grid, distance_grid: Grid axes
        - data_table: dict of phase → 2D array [depth, distance]
        - interp_1d_func: function(phase, depth_index, distance) → float

        Returns:
        - interpolated value
        """
        if self.depth_grid is None or self.distance_grid is None:
            raise ValueError("Interpolation grid not initialized.")

        # Clamp values
        depth = max(min(depth, self.depth_grid[-1]), self.depth_grid[0])
        distance = max(min(distance, self.distance_grid[-1]), self.distance_grid[0])

        # Locate depth interval
        i1 = int(np.searchsorted(self.depth_grid, depth))
        i0 = i1 - 1
        if i0 < 0:
            i0 = 0
        if i1 >= len(self.depth_grid):
            i1 = len(self.depth_grid) - 1
        if i0 == i1:
            return interp_1d_func(phase, i0, distance)

        # Interpolate at both depths
        val0 = interp_1d_func(phase, i0, distance)
        val1 = interp_1d_func(phase, i1, distance)
        d0, d1 = self.depth_grid[i0], self.depth_grid[i1]
        if d1 == d0:
            return val0
        frac = (depth - d0) / (d1 - d0)
        return val0 + frac * (val1 - val0)

    def interpolate_travel_time(
        self, phase: str, depth: float, distance: float
    ) -> float:
        """
        Interpolate the travel time (seconds) for a given phase, source depth (km), and distance (km) from the precomputed table.
        """
        return self.interpolate_2d_slice(
            phase,
            depth,
            distance,
            self._interp_time_one_depth,
        )

    def interpolate_takeoff_angle(
        self, phase: str, depth: float, distance: float
    ) -> float:
        """
        Interpolate the takeoff angle (degrees) from the precomputed table.
        """
        return self.interpolate_2d_slice(
            phase,
            depth,
            distance,
            self._interp_takeoff_angle_one_depth,
        )

    def interpolate_ray_param(self, phase: str, depth: float, distance: float) -> float:
        """
        Interpolate the ray parameter (s/radian) for a given phase, source depth, and distance from the precomputed table.
        """
        return self.interpolate_2d_slice(
            phase,
            depth,
            distance,
            self._interp_ray_param_one_depth,
        )

    def _interp_time_one_depth(
        self, phase: str, depth_index: int, distance: float
    ) -> float:
        return adaptive_interp_1d(
            distance, self.distance_grid, self.travel_time_tables[phase][depth_index]
        )

    def _interp_takeoff_angle_one_depth(
        self, phase: str, depth_index: int, distance: float
    ) -> float:
        return adaptive_interp_1d(
            distance, self.distance_grid, self.takeoff_angle_tables[phase][depth_index]
        )

    def _interp_ray_param_one_depth(
        self, phase: str, depth_index: int, distance: float
    ) -> float:
        return adaptive_interp_1d(
            distance, self.distance_grid, self.ray_param_tables[phase][depth_index]
        )
