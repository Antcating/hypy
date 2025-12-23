"""
travel_time_tables.py
--------------------
Handles travel-time, ray-parameter, and takeoff angle table creation, saving, loading, and interpolation.
"""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from obspy.taup import TauPyModel  # type: ignore[import-untyped]

from hypy.logger import logger
from hypy.propagation import PropagationModel
from hypy.utils.geo import get_spatial_slowness_1d, gps2dist_azimuth_nb
from hypy.utils.interp import interp_quant


class TauTimeTables(PropagationModel):
    """
    Manages travel-time, ray-parameter, and takeoff angle tables for seismic phases.
    """

    def __init__(
        self,
        model_tvel: str,
        model_name: str,
        phases: List[str],
        max_depth: float,
        max_distance: float,
        depth_step: float,
        distance_step: float,
        engine: str = "pyrocko",
        overwrite: bool = False,
    ):
        self.model = TauPyModel(model_name)
        self.engine = engine
        self.tracer: PropagationModel
        if engine == "taup":
            from hypy.propagation.ray_tracers.taup import TauPRayTracer

            self.tracer = TauPRayTracer(model_tvel)
        elif engine == "pyrocko":
            from hypy.propagation.ray_tracers.pyrocko import PyrockoRayTracer

            self.tracer = PyrockoRayTracer(model_tvel)
        else:
            raise ValueError(f"Unsupported engine: {engine}")
        self.model_name = model_name
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

    def _table_filename(self, phase) -> str:
        table_root_dir = Path(__file__).parent
        table_name = f"travel_time_tables_{self.model_name}_{self.engine}_{phase}_d{self.max_depth}_r{self.max_distance}_dz{self.depth_step}_dr{self.distance_step}.npz"
        return os.path.join(table_root_dir, "data", table_name)

    def save(self, filename) -> None:
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
        np.savez_compressed(filename, **save_dict)

    def load(self, filename: str, phase: str) -> None:
        """
        Load travel-time, ray-parameter, and takeoff angle tables from a file.
        """
        data = np.load(filename, allow_pickle=True)
        self.depth_grid = data["depth_grid"]
        self.distance_grid = data["distance_grid"]
        if f"tt_{phase}" in data:
            self.travel_time_tables[phase] = data[f"tt_{phase}"]
        if f"rp_{phase}" in data:
            self.ray_param_tables[phase] = data[f"rp_{phase}"]
        if f"ta_{phase}" in data:
            self.takeoff_angle_tables[phase] = data[f"ta_{phase}"]

    def _build_tables(self, overwrite: bool = False) -> None:
        """
        Build (or load) the travel-time, ray-parameter, and takeoff angle tables for the given phases.
        """
        for phase in self.phases:
            fname = self._table_filename(phase)
            if os.path.exists(fname) and not overwrite:
                logger.info(f"Loading travel-time tables from {fname}")
                self.load(fname, phase)
                continue
            depth_values = np.arange(0.1, self.max_depth + 0.1 + 1e-6, self.depth_step)
            distance_values = np.arange(
                0.1, self.max_distance + 0.1 + 1e-6, self.distance_step
            )
            self.depth_grid = depth_values
            self.distance_grid = distance_values

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
                    # Handle P phase
                    if phase == "P" and "P" in self.travel_time_tables:
                        time, angle = self.tracer.get_arrival(depth, dist, "P")
                        self.travel_time_tables["P"][i, j] = time
                        self.takeoff_angle_tables["P"][i, j] = angle
                    # Handle S phase
                    elif phase == "S" and "S" in self.travel_time_tables:
                        time, angle = self.tracer.get_arrival(depth, dist, "S")
                        self.travel_time_tables["S"][i, j] = time
                        self.takeoff_angle_tables["S"][i, j] = angle
            logger.info(f"Saving travel-time tables to {fname}")
            self.save(fname)

    def compute_travel_time(self, phase: str, depth: float, distance: float) -> float:
        return interp_quant(
            depth,
            distance,
            self.depth_grid,
            self.distance_grid,
            self.travel_time_tables[phase],
        )

    def compute_takeoff_angle(self, phase: str, depth: float, distance: float) -> float:
        return interp_quant(
            depth,
            distance,
            self.depth_grid,
            self.distance_grid,
            self.takeoff_angle_tables[phase],
        )

    def get_arrival(self, source_depth, distance, phase):
        if phase not in self.phases:
            raise ValueError(f"Phase {phase} not in available phases: {self.phases}")
        time = self.compute_travel_time(phase, source_depth, distance)
        takeoff_angle = self.compute_takeoff_angle(phase, source_depth, distance)
        if np.isnan(time) or np.isnan(takeoff_angle):
            raise ValueError(
                f"No travel time or takeoff angle found for phase {phase} at depth {source_depth} and distance {distance}."
            )
        return time, takeoff_angle

    def calculate_arrival_parameters(
        self,
        source_lon: float,
        source_lat: float,
        source_depth_km: float,
        receiver_lon: float,
        receiver_lat: float,
        receiver_elev_m: float,
        phase: str,
    ) -> "tuple[float, float, tuple[float, float, float]]":
        """
        Get arrival parameters for all phases between source and receiver.
        """
        dist_m, azimuth_deg, _ = gps2dist_azimuth_nb(
            source_lat,
            source_lon,
            receiver_lat,
            receiver_lon,
            R=self.model.model.radius_of_planet * 1000.0,
        )
        distance_km = dist_m / 1000.0

        source_depth_km += receiver_elev_m / 1000.0  # account for station elevation

        travel_time, takeoff_angle = self.get_arrival(
            source_depth=source_depth_km, distance=distance_km, phase=phase
        )
        # Local velocity and partial derivatives
        velocity_km_s = self.model.model.s_mod.v_mod.evaluate_below(
            source_depth_km, phase
        )[0]
        takeoff_rad = np.deg2rad(takeoff_angle)

        # dt/dr (ray), dt/dx (east), dt/dy (north), dt/dz (down)
        _, dt_dx, dt_dy, dt_dz = get_spatial_slowness_1d(
            takeoff_rad, azimuth_deg, velocity_km_s
        )

        return distance_km, travel_time, (dt_dx, dt_dy, dt_dz)

    def calculate_arrival_parameters_bulk(
        self,
        source_lon: float,
        source_lat: float,
        source_depth_km: float,
        receiver_lons: List[float],
        receiver_lats: List[float],
        receiver_elevs_m: List[float],
        phase: str,
    ):
        """
        Get arrival parameters for all phases between source and multiple receivers.
        """
        results = []
        for receiver_lon, receiver_lat, receiver_elev_m in zip(
            receiver_lons, receiver_lats, receiver_elevs_m
        ):
            result = self.calculate_arrival_parameters(
                source_lon,
                source_lat,
                source_depth_km,
                receiver_lon,
                receiver_lat,
                receiver_elev_m,
                phase,
            )
            results.append(result)
        return results
