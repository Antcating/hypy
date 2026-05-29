"""Precomputed 1D travel-time tables handler for layered Earth models using t-p.

This module provides the TauTravelTimeTable class, which manages precomputed
travel-time, ray-parameter, and takeoff angle tables for seismic phases
based on a 1D velocity model.

It supports both Pyrocko and TauP backends for generating these tables,
allowing for efficient retrieval of travel times and ray parameters
for given source-receiver configurations and seismic phases.

Returns
-------
TauTimeTables
    A class that manages precomputed travel-time tables for seismic phases
    based on a 1D velocity model, supporting both Pyrocko and TauP backends.

Raises
------
ValueError
    Unsupported engine specified for table generation.
ValueError
    Phase not in available phases when requesting arrival parameters.
ValueError
    No travel time or takeoff angle found for given phase, depth, and distance.
ValueError
    Station at distance exceeding max_distance when requesting arrival parameters.
ValueError
    Travel-time tables not built when attempting to save tables to file.
ValueError
    Phase not in available phases when attempting to load tables from file.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from hypy.logger import logger
from hypy.propagation.layered import LayeredPropagationEngine
from hypy.utils.geo import gps2dist_azimuth_nb
from hypy.utils.math import interp_quant


class TauTravelTimeGrid(LayeredPropagationEngine):
    """Tau travel-time table for layered Earth models.

    This class manages precomputed travel-time, ray-parameter, and takeoff angle
    tables for seismic phases based on a 1D velocity model.
    It supports both Pyrocko and TauP backends for generating these tables,
    allowing for efficient retrieval of travel times and ray parameters for given
    source-receiver configurations and seismic phases.

    Parameters
    ----------
    model_tvel : str
        Path to the velocity model file (only .tvel files are supported).
    phases : List[str]
        A list of seismic phases for which to compute travel times.
    max_depth : float
        The maximum depth for the travel-time tables.
    max_distance : float
        The maximum distance for the travel-time tables.
    depth_step : float
        The step size for the depth grid.
    distance_step : float
        The step size for the distance grid.
    engine : str, optional
        The backend to use for generating the travel-time tables. by default "pyrocko"
    overwrite : bool, optional
        Whether to overwrite existing travel-time tables. by default False
    """  # noqa: E501

    def __init__(
        self,
        model_tvel: str,
        phases: List[str],
        max_depth: float,
        max_distance: float,
        depth_step: float,
        distance_step: float,
        engine: str = "pyrocko",
        overwrite: bool = False,
    ):
        super().__init__(model_tvel)
        model_name = Path(model_tvel).stem
        self.backend: LayeredPropagationEngine
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
        self.load_tables(model_name=model_name, engine=engine, overwrite=overwrite)

    def _table_filename(self, model_name: str, engine: str, phase: str) -> str:
        table_root_dir = Path(__file__).parent
        table_name = f"travel_time_tables_{model_name}_{engine}_{phase}_d{self.max_depth}_r{self.max_distance}_dz{self.depth_step}_dr{self.distance_step}.npz"  # noqa: E501
        return os.path.join(table_root_dir, "data", table_name)

    def save(self, filename: str | Path) -> None:
        """Save the travel-time, ray-parameter, and takeoff angle tables to a file.

        The file is saved in compressed NumPy format (.npz) and contains the
        depth grid, distance grid, and the tables for each phase.

        TODO: Consider adding metadata to the saved file,
        such as model name, engine, phase, and grid parameters,
        to make it easier to identify and manage multiple tables.

        TODO: Consider saving tables in a more structured format (e.g., HDF5)
        that can store metadata and allow for more efficient access to specific phases
        or parameters without loading the entire file into memory.

        The filename is generated based on the model name, engine, phase.

        Parameters
        ----------
        filename : str | Path
            The path to the file where the tables should be saved.

        Raises
        ------
        ValueError
            If the travel-time tables have not been built and there is nothing to save.
        """
        if self.depth_grid is None or self.distance_grid is None:
            raise ValueError("Travel-time tables are not built; nothing to save.")
        save_dict = {"depth_grid": self.depth_grid, "distance_grid": self.distance_grid}
        if (
            self.travel_time_tables is None
            or self.travel_time_tables == {}
            or self.ray_param_tables is None
            or self.ray_param_tables == {}
            or self.takeoff_angle_tables is None
            or self.takeoff_angle_tables == {}
        ):
            raise ValueError("Travel-time tables are not built; nothing to save.")
        for ph in self.travel_time_tables:
            save_dict[f"tt_{ph}"] = self.travel_time_tables[ph]
        for ph in self.ray_param_tables:
            save_dict[f"rp_{ph}"] = self.ray_param_tables[ph]
        for ph in self.takeoff_angle_tables:
            save_dict[f"ta_{ph}"] = self.takeoff_angle_tables[ph]
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        np.savez_compressed(filename, allow_pickle=True, **save_dict)

    def load_from_file(self, filename: str | Path, phase: str) -> None:
        """Load the tables for a specific phase from a file.

        Parameters
        ----------
        filename : str | Path
            The path to the file from which to load the tables.
        phase : str
            The seismic phase for which to load the tables.

        Raises
        ------
        ValueError
            If the travel-time tables are not found in the file for the given phase.
        ValueError
            If the ray-parameter tables are not found in the file for the given phase.
        ValueError
            If the takeoff angle tables are not found in the file for the given phase.
        """
        data = np.load(filename, allow_pickle=True)
        self.depth_grid = data["depth_grid"]
        self.distance_grid = data["distance_grid"]
        if f"tt_{phase}" in data:
            self.travel_time_tables[phase] = data[f"tt_{phase}"]
        else:
            raise ValueError(
                f"Phase {phase} not found in travel-time tables in file {filename}."
            )
        if f"rp_{phase}" in data:
            self.ray_param_tables[phase] = data[f"rp_{phase}"]
        else:
            raise ValueError(
                f"Phase {phase} not found in ray-parameter tables in file {filename}."
            )
        if f"ta_{phase}" in data:
            self.takeoff_angle_tables[phase] = data[f"ta_{phase}"]
        else:
            raise ValueError(
                f"Phase {phase} not found in takeoff angle tables in file {filename}."
            )

    def load_tables(
        self, model_name: str, engine: str, overwrite: bool = False
    ) -> None:
        """Load or build the travel-time, ray-parameter, and takeoff angle tables for all phases.

        Parameters
        ----------
        model_name : str
            The name of the seismic model.
        engine : str
            The engine to use for table generation.
        overwrite : bool, optional
            Whether to overwrite existing table files, by default False

        Raises
        ------
        ValueError
            If an unsupported engine is specified for table generation.
        """  # noqa: E501
        for phase in self.phases:
            fname = self._table_filename(model_name, engine, phase)
            if os.path.exists(fname) and not overwrite:
                logger.info(f"Loading travel-time tables from {fname}")
                self.load_from_file(fname, phase)
                continue
            logger.info(
                f"Building travel-time tables for phase {phase} using engine {engine}"
            )

            # Initialize tracer for table generation
            if engine == "pyrocko":
                from hypy.propagation.layered.ray_tracer.pyrocko import PyrockoRayTracer

                self.tracer = PyrockoRayTracer(self.model_path)
            elif engine == "taup":
                from hypy.propagation.layered.ray_tracer.taup import TauPRayTracer

                self.tracer = TauPRayTracer(self.model_path)
            else:
                raise ValueError(f"Unsupported engine: {engine}")

            # FIXME: Depth values should start from 0, not from 0.1,
            # but TauP and Pyrocko do not return valid travel times for
            # very shallow depths (e.g., 0 km),
            # so we start from a small positive value (0.1 km) to avoid
            # issues with table generation. This is a workaround and should be revisited
            # in the future to ensure that the tables can cover the full depth range
            # starting from 0 km if needed.
            depth_values = np.arange(0.1, self.max_depth + 0.1 + 1e-6, self.depth_step)
            # FIXME: Distance values should start from 0, not from 0.1,
            # but TauP and Pyrocko do not return valid travel times for
            # very short distances (e.g., 0 km),
            # so we start from a small positive value (0.1 km) to avoid issues with
            # table generation. This is a workaround and should be revisited
            # in the future to ensure that the tables can cover the full distance range
            # starting from 0 km if needed.
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
                    if phase in self.travel_time_tables:
                        time, angle = self.tracer.get_arrival_1d(dist, depth, phase)
                        self.travel_time_tables[phase][i, j] = time
                        self.takeoff_angle_tables[phase][i, j] = angle
            logger.info(f"Saving travel-time tables to {fname}")
            self.save(fname)

    def compute_travel_time(
        self, distance_km: float, depth_km: float, phase: str
    ) -> float:
        """Compute the travel time for a parameters by interpolating values of tables.

        Uses the precomputed travel-time tables to interpolate the travel time
        for a given distance, depth, and seismic phase.

        The interpolation is performed using the generic interpolation function that
        can handle both linear and Lagrange interpolation based on the proximity of the
        query point to the grid points.

        Parameters
        ----------
        distance_km : float
            The distance between the source and receiver in kilometers.
        depth_km : float
            The depth of the source in kilometers.
        phase : str
            The seismic phase.

        Returns
        -------
        float
            The computed travel time.
        """
        return interp_quant(
            depth_km,
            distance_km,
            self.depth_grid,
            self.distance_grid,
            self.travel_time_tables[phase],
        )

    def compute_takeoff_angle(
        self, distance_km: float, depth_km: float, phase: str
    ) -> float:
        """Compute the takeoff angle for a parameters by interpolating values of tables.

        Uses the precomputed takeoff angle tables to interpolate the takeoff angle
        for a given distance, depth, and seismic phase.

        The interpolation is performed using the generic interpolation function that
        can handle both linear and Lagrange interpolation based on the proximity of the
        query point to the grid points.

        Parameters
        ----------
        distance_km : float
            The distance between the source and receiver in kilometers.
        depth_km : float
            The depth of the source in kilometers.
        phase : str
            The seismic phase.

        Returns
        -------
        float
            The computed takeoff angle.
        """
        return interp_quant(
            depth_km,
            distance_km,
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
        """Get arrival parameters for a given source-receiver configuration.

        The arrival parameters include the arc distance between source and receiver,
        the arrival time, the takeoff angle, and the azimuth from source to receiver.

        Parameters
        ----------
        source_latitude : float
            Latitude of the seismic source in degrees.
        source_longitude : float
            Longitude of the seismic source in degrees.
        source_depth_km : float
            Depth of the seismic source in kilometers.
        receiver_latitude : float
            Latitude of the seismic receiver in degrees.
        receiver_longitude : float
            Longitude of the seismic receiver in degrees.
        receiver_elevation_m : float
            Elevation of the seismic receiver in meters.
        phase : str
            The seismic phase.
        max_distance_km : float | None, optional
            The maximum distance for which to compute arrivals, by default None

        Returns
        -------
        tuple[float, Optional[float], Optional[float], float]
            A tuple containing:

            arc_distance_km : float
                Great-circle distance between source and receiver in kilometers.
            arrival_time : float or None
                Arrival time in seconds, or None if (distance >= max_distance_km).
            takeoff_angle : float or None
                Takeoff angle in degrees, or None if (distance >= max_distance_km).
            azimuth_deg : float
                Azimuth from source to receiver in degrees (clockwise from North).

        Raises
        ------
        ValueError
            If the specified phase is not in the available phases.
        ValueError
            If no travel time or takeoff angle is found for the given configuration.
        """
        if phase not in self.phases:
            raise ValueError(f"Phase {phase} not in available phases: {self.phases}")

        dist_m, azimuth_deg, _ = gps2dist_azimuth_nb(
            source_latitude,
            source_longitude,
            receiver_latitude,
            receiver_longitude,
            R=self.velocity_model.radius_of_planet_km * 1000.0,
        )
        arc_distance_km = dist_m / 1000.0

        # Check max distance
        # This check is nessesary when using Tables with limited grid coverage
        if max_distance_km is not None and arc_distance_km > max_distance_km:
            logger.info(
                "Station at distance %.2f km exceeds max_distance %.2f km. Skipping.",
                arc_distance_km,
                max_distance_km,
            )
            return arc_distance_km, None, None, azimuth_deg

        source_depth_km += receiver_elevation_m / 1000.0

        time = self.compute_travel_time(arc_distance_km, source_depth_km, phase)
        takeoff_angle = self.compute_takeoff_angle(
            arc_distance_km, source_depth_km, phase
        )
        if np.isnan(time) or np.isnan(takeoff_angle):
            raise ValueError(
                f"No travel time or takeoff angle found for phase {phase}"
                f" at depth {source_depth_km} and distance {arc_distance_km}."
            )
        return arc_distance_km, time, takeoff_angle, azimuth_deg

    def get_arrival_1d(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ) -> tuple[float, float]:
        """Get the arrival time and takeoff angle for a given configuration.

        Layered 1D models are symmetric, so the specific source and receiver locations
        do not matter, only the distance between them (arc_distance_km) and the source
        depth (source_depth_km) matter for computing the arrival time and takeoff angle
        for a given seismic phase.

        Parameters
        ----------
        arc_distance_km : float
            The distance between the source and receiver in kilometers.
        source_depth_km : float
            The depth of the source in kilometers.
        phase : str
            The seismic phase for which to compute arrivals.

        Returns
        -------
        tuple[float, float]
            Tuple with the arrival time in seconds and the takeoff angle in degrees.

        Raises
        ------
        ValueError
            If the specified phase is not in the available phases.
        ValueError
            If no travel time or takeoff angle is found for the given configuration.
        """
        if phase not in self.phases:
            raise ValueError(f"Phase {phase} not in available phases: {self.phases}")
        # Normalize distrance to positive values (valid for 1D models)
        arc_distance_km = abs(arc_distance_km)
        time = self.compute_travel_time(arc_distance_km, source_depth_km, phase)
        takeoff_angle = self.compute_takeoff_angle(
            arc_distance_km, source_depth_km, phase
        )
        if np.isnan(time) or np.isnan(takeoff_angle):
            raise ValueError(
                f"No travel time or takeoff angle found for phase {phase}"
                f" at depth {source_depth_km} and distance {arc_distance_km}."
            )
        return time, takeoff_angle
