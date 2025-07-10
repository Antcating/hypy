"""
travel_time_tables.py
--------------------
Handles travel-time, ray-parameter, and takeoff angle table creation, saving, loading, and interpolation.
"""

from typing import Dict, List, Optional
import numpy as np
from obspy.taup import TauPyModel
from obspy.geodetics import kilometer2degrees
import os
from pathlib import Path


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

    def interpolate_travel_time(
        self, phase: str, depth: float, distance: float
    ) -> float:
        """
        Interpolate the travel time (seconds) for a given phase, source depth (km), and distance (km) from the precomputed table.
        """
        if self.depth_grid is None or self.distance_grid is None:
            raise ValueError("Travel-time tables not initialized.")
        depth = max(min(depth, self.depth_grid[-1]), self.depth_grid[0])
        distance = max(min(distance, self.distance_grid[-1]), self.distance_grid[0])
        i1 = int(np.searchsorted(self.depth_grid, depth))
        i0 = int(i1 - 1)
        if i0 < 0:
            i0 = 0
        if i1 >= len(self.depth_grid):
            i1 = len(self.depth_grid) - 1
        if i0 == i1:
            return self._interp_time_one_depth(phase, int(i0), distance)
        t0 = self._interp_time_one_depth(phase, int(i0), distance)
        t1 = self._interp_time_one_depth(phase, int(i1), distance)
        depth0, depth1 = self.depth_grid[i0], self.depth_grid[i1]
        if depth1 == depth0:
            return t0
        frac = (depth - depth0) / (depth1 - depth0)
        return t0 + frac * (t1 - t0)

    def interpolate_takeoff_angle(
        self, phase: str, depth: float, distance: float
    ) -> float:
        """
        Interpolate the takeoff angle (degrees) from the precomputed table.
        """
        if self.depth_grid is None or self.distance_grid is None:
            raise ValueError("Takeoff angle table is not initialized.")
        depth = max(min(depth, self.depth_grid[-1]), self.depth_grid[0])
        distance = max(min(distance, self.distance_grid[-1]), self.distance_grid[0])
        i1 = int(np.searchsorted(self.depth_grid, depth))
        i0 = int(i1 - 1)
        if i0 < 0:
            i0 = 0
        if i1 >= len(self.depth_grid):
            i1 = len(self.depth_grid) - 1
        if i0 == i1:
            return self._interp_takeoff_angle_one_depth(phase, int(i0), distance)
        t0 = self._interp_takeoff_angle_one_depth(phase, int(i0), distance)
        t1 = self._interp_takeoff_angle_one_depth(phase, int(i1), distance)
        d0, d1 = self.depth_grid[i0], self.depth_grid[i1]
        if d1 == d0:
            return t0
        return t0 + (depth - d0) / (d1 - d0) * (t1 - t0)

    def interpolate_ray_param(self, phase: str, depth: float, distance: float) -> float:
        """
        Interpolate the ray parameter (s/radian) for a given phase, source depth, and distance from the precomputed table.
        """
        if self.depth_grid is None or self.distance_grid is None:
            raise ValueError("Travel-time tables not initialized.")
        depth = max(min(depth, self.depth_grid[-1]), self.depth_grid[0])
        distance = max(min(distance, self.distance_grid[-1]), self.distance_grid[0])
        i1 = int(np.searchsorted(self.depth_grid, depth))
        i0 = int(i1 - 1)
        if i0 < 0:
            i0 = 0
        if i1 >= len(self.depth_grid):
            i1 = len(self.depth_grid) - 1
        if i0 == i1:
            return self._interp_ray_param_one_depth(phase, int(i0), distance)
        p0 = self._interp_ray_param_one_depth(phase, int(i0), distance)
        p1 = self._interp_ray_param_one_depth(phase, int(i1), distance)
        depth0, depth1 = self.depth_grid[i0], self.depth_grid[i1]
        if depth1 == depth0:
            return p0
        frac = (depth - depth0) / (depth1 - depth0)
        return p0 + frac * (p1 - p0)

    def _interp_time_one_depth(
        self, phase: str, depth_index: int, distance: float
    ) -> float:
        times_row = self.travel_time_tables[phase][depth_index]
        dist_grid = self.distance_grid
        if distance <= dist_grid[0]:
            return times_row[0]
        if distance >= dist_grid[-1]:
            j0 = len(dist_grid) - 2
            j1 = len(dist_grid) - 1
            x0, x1 = dist_grid[j0], dist_grid[j1]
            y0, y1 = times_row[j0], times_row[j1]
            if x1 == x0:
                return y1
            return y1 + (distance - x1) * (y1 - y0) / (x1 - x0)
        j1 = np.searchsorted(dist_grid, distance)
        j0 = j1 - 1
        if j0 < 0:
            j0 = 0
            j1 = 0
        if j1 >= len(dist_grid):
            j1 = len(dist_grid) - 1
            j0 = j1
        if dist_grid[j0] == distance or j0 == j1:
            return times_row[j0]
        if j0 >= 1 and j1 < len(dist_grid) - 1:
            if (distance - dist_grid[j0]) < (dist_grid[j1] - distance):
                idxs = [j0 - 1, j0, j1]
            else:
                idxs = [j0, j1, j1 + 1]
        else:
            idxs = [j0, j1]
        if len(idxs) == 2:
            x0, x1 = dist_grid[idxs[0]], dist_grid[idxs[1]]
            y0, y1 = times_row[idxs[0]], times_row[idxs[1]]
            if x1 == x0:
                return y0
            return y0 + (distance - x0) / (x1 - x0) * (y1 - y0)
        jA, jB, jC = idxs
        xA, xB, xC = dist_grid[jA], dist_grid[jB], dist_grid[jC]
        yA, yB, yC = times_row[jA], times_row[jB], times_row[jC]
        denomA = (xA - xB) * (xA - xC)
        denomB = (xB - xA) * (xB - xC)
        denomC = (xC - xA) * (xC - xB)
        termA = yA * (distance - xB) * (distance - xC) / denomA
        termB = yB * (distance - xA) * (distance - xC) / denomB
        termC = yC * (distance - xA) * (distance - xB) / denomC
        return termA + termB + termC

    def _interp_takeoff_angle_one_depth(
        self, phase: str, depth_index: int, distance: float
    ) -> float:
        angle_row = self.takeoff_angle_tables[phase][depth_index]
        dist_grid = self.distance_grid
        if distance <= dist_grid[0]:
            return angle_row[0]
        if distance >= dist_grid[-1]:
            j0 = len(dist_grid) - 2
            j1 = len(dist_grid) - 1
            x0, x1 = dist_grid[j0], dist_grid[j1]
            y0, y1 = angle_row[j0], angle_row[j1]
            return y1 + (distance - x1) * (y1 - y0) / (x1 - x0)
        j1 = np.searchsorted(dist_grid, distance)
        j0 = j1 - 1
        if j0 < 0:
            j0 = 0
        if j1 >= len(dist_grid):
            j1 = len(dist_grid) - 1
        if dist_grid[j0] == distance or j0 == j1:
            return angle_row[j0]
        if j0 >= 1 and j1 < len(dist_grid) - 1:
            if (distance - dist_grid[j0]) < (dist_grid[j1] - distance):
                idxs = [j0 - 1, j0, j1]
            else:
                idxs = [j0, j1, j1 + 1]
        else:
            idxs = [j0, j1]
        if len(idxs) == 2:
            x0, x1 = dist_grid[idxs[0]], dist_grid[idxs[1]]
            y0, y1 = angle_row[idxs[0]], angle_row[idxs[1]]
            return y0 + (distance - x0) / (x1 - x0) * (y1 - y0)
        jA, jB, jC = idxs
        xA, xB, xC = dist_grid[jA], dist_grid[jB], dist_grid[jC]
        yA, yB, yC = angle_row[jA], angle_row[jB], angle_row[jC]
        denomA = (xA - xB) * (xA - xC)
        denomB = (xB - xA) * (xB - xC)
        denomC = (xC - xA) * (xC - xB)
        return (
            yA * (distance - xB) * (distance - xC) / denomA
            + yB * (distance - xA) * (distance - xC) / denomB
            + yC * (distance - xA) * (distance - xB) / denomC
        )

    def _interp_ray_param_one_depth(
        self, phase: str, depth_index: int, distance: float
    ) -> float:
        params_row = self.ray_param_tables[phase][depth_index]
        dist_grid = self.distance_grid
        if distance <= dist_grid[0]:
            return params_row[0]
        if distance >= dist_grid[-1]:
            j0 = len(dist_grid) - 2
            j1 = len(dist_grid) - 1
            x0, x1 = dist_grid[j0], dist_grid[j1]
            y0, y1 = params_row[j0], params_row[j1]
            if x1 == x0:
                return y1
            return y1 + (distance - x1) * (y1 - y0) / (x1 - x0)
        j1 = np.searchsorted(dist_grid, distance)
        j0 = j1 - 1
        if j0 < 0:
            j0 = 0
            j1 = 0
        if j1 >= len(dist_grid):
            j1 = len(dist_grid) - 1
            j0 = j1
        if dist_grid[j0] == distance or j0 == j1:
            return params_row[j0]
        if j0 >= 1 and j1 < len(dist_grid) - 1:
            if (distance - dist_grid[j0]) < (dist_grid[j1] - distance):
                idxs = [j0 - 1, j0, j1]
            else:
                idxs = [j0, j1, j1 + 1]
        else:
            idxs = [j0, j1]
        if len(idxs) == 2:
            x0, x1 = dist_grid[idxs[0]], dist_grid[idxs[1]]
            y0, y1 = params_row[idxs[0]], params_row[idxs[1]]
            if x1 == x0:
                return y0
            return y0 + (distance - x0) / (x1 - x0) * (y1 - y0)
        jA, jB, jC = idxs
        xA, xB, xC = dist_grid[jA], dist_grid[jB], dist_grid[jC]
        yA, yB, yC = params_row[jA], params_row[jB], params_row[jC]
        denomA = (xA - xB) * (xA - xC)
        denomB = (xB - xA) * (xB - xC)
        denomC = (xC - xA) * (xC - xB)
        termA = yA * (distance - xB) * (distance - xC) / denomA
        termB = yB * (distance - xA) * (distance - xC) / denomB
        termC = yC * (distance - xA) * (distance - xB) / denomC
        return termA + termB + termC

