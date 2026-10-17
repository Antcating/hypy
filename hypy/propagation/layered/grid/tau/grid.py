"""Precomputed 1D travel-time tables handler for layered Earth models using TauP.

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
from typing import Dict, List

import numpy as np

from hypy.logger import logger
from hypy.model import VelocityModel
from hypy.propagation.layered.grid import LayeredGrid1D, model_fingerprint
from hypy.propagation.layered.ray_tracer import LayeredRayTracer
from hypy.utils.seismic import normalize_phase

_TABLE_CACHE_VERSION = "v3"


def _grid_nodes(start: float, stop: float, step: float) -> np.ndarray:
    """Return nodes from ``start`` to ``stop`` inclusive."""
    nodes = np.arange(start, stop + 1e-6, step)
    if nodes[-1] < stop:
        nodes = np.append(nodes, stop)
    return nodes


class TauTravelTimeGrid(LayeredGrid1D):
    """Tau travel-time table for layered Earth models.

    This class manages precomputed travel-time, ray-parameter, and takeoff angle
    tables for seismic phases based on a 1D velocity model.
    It supports both Pyrocko and TauP backends for generating these tables,
    allowing for efficient retrieval of travel times and ray parameters for given
    source-receiver configurations and seismic phases.

    Parameters
    ----------
    model : VelocityModel
        The velocity model (only .tvel models are supported).
    phases : List[str]
        A list of seismic phases for which to compute travel times.
    max_depth_km : float
        The maximum depth for the travel-time tables.
    max_distance_km : float
        The maximum distance for the travel-time tables.
    depth_step_km : float
        The step size for the depth grid.
    distance_step_km : float
        The step size for the distance grid.
    tracer : LayeredRayTracer
        An instantiated layered ray tracer (e.g. ``TauPRayTracer`` or
        ``PyrockoRayTracer``) used to generate the travel-time tables.
    overwrite : bool, optional
        Whether to overwrite existing travel-time tables. by default False
    """  # noqa: E501

    feature_flags = frozenset(
        {
            "propagation.layered",
            "propagation.table",
            "derivative.analytical",
            "derivative.numerical",
            "derivative.batch",
        }
    )

    required_model_features = frozenset({"model.profile.1d"})

    def __init__(
        self,
        model: VelocityModel,
        phases: List[str],
        max_depth_km: float,
        max_distance_km: float,
        depth_step_km: float,
        distance_step_km: float,
        tracer: LayeredRayTracer,
        overwrite: bool = False,
    ):
        super().__init__(model)
        model_name = Path(model.model_path).stem
        self.tracer: LayeredRayTracer = tracer
        if not callable(getattr(self.tracer, "get_arrival_1d", None)):
            raise ValueError(
                f"Tracer {tracer.name} does not support get_arrival_1d method. "
                "Please use a tracer that supports get_arrival_1d method."
            )
        self.phases = [normalize_phase(phase) for phase in phases]
        self.max_depth_km = max_depth_km
        self.max_distance_km = max_distance_km
        self.depth_step_km = depth_step_km
        self.distance_step_km = distance_step_km
        self.depth_grid: np.ndarray
        self.distance_grid: np.ndarray
        self.travel_time_tables: Dict[str, np.ndarray] = {}
        self.takeoff_angle_tables: Dict[str, np.ndarray] = {}

        self.load_tables(model_name=model_name, tracer=tracer, overwrite=overwrite)

    def _table_filename(self, model_name: str, engine: str, phase: str) -> str:
        table_root_dir = Path(__file__).parent
        model_hash = model_fingerprint(self.velocity_model)
        table_name = (
            f"travel_time_tables_{model_name}_{model_hash}_{_TABLE_CACHE_VERSION}"
            f"_{engine}_{phase}_d{self.max_depth_km}_r{self.max_distance_km}"
            f"_dz{self.depth_step_km}_dr{self.distance_step_km}.npz"
        )
        return os.path.join(table_root_dir, "data", table_name)

    def load_tables(
        self, model_name: str, tracer: LayeredRayTracer, overwrite: bool = False
    ) -> None:
        """Load or build the travel-time and takeoff angle tables for all phases.

        Parameters
        ----------
        model_name : str
            The name of the seismic model.
        tracer : LayeredRayTracer
            The engine to use for table generation.
        overwrite : bool, optional
            Whether to overwrite existing table files, by default False

        Raises
        ------
        ValueError
            If an unsupported tracer is specified for table generation.
        """  # noqa: E501
        for phase in self.phases:
            fname = self._table_filename(model_name, tracer.name, phase)
            if os.path.exists(fname) and not overwrite:
                logger.info(f"Loading travel-time tables from {fname}")
                self.load(fname, phase)
                continue
            logger.info(
                f"Building travel-time tables for phase {phase} using tracer {tracer.name}"  # noqa: E501
            )

            # FIXME: Depth values should start from 0, not from 0.1,
            # but TauP and Pyrocko do not return valid travel times for
            # very shallow depths (e.g., 0 km),
            # so we start from a small positive value (0.1 km) to avoid
            # issues with table generation. This is a workaround and should be revisited
            # in the future to ensure that the tables can cover the full depth range
            # starting from 0 km if needed.
            depth_values = _grid_nodes(0.1, self.max_depth_km, self.depth_step_km)
            # FIXME: Distance values should start from 0, not from 0.1,
            # but TauP and Pyrocko do not return valid travel times for
            # very short distances (e.g., 0 km),
            # so we start from a small positive value (0.1 km) to avoid issues with
            # table generation. This is a workaround and should be revisited
            # in the future to ensure that the tables can cover the full distance range
            # starting from 0 km if needed.
            distance_values = _grid_nodes(
                0.1, self.max_distance_km, self.distance_step_km
            )
            self.depth_grid = depth_values
            self.distance_grid = distance_values

            self.travel_time_tables[phase] = np.full(
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
