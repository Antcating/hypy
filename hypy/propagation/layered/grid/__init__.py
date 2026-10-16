"""Layered travel-time grid backends."""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numba import njit, prange

from hypy.logger import logger
from hypy.model import VelocityModel
from hypy.propagation.layered import LayeredPropagationEngine
from hypy.utils.geo import (
    get_analytical_derivative_1d_batch,
    gps2dist_azimuth_batch_nb,
)
from hypy.utils.math.interpolation import interp_quant, interp_quant_batch
from hypy.utils.seismic import normalize_phase


def model_fingerprint(model) -> str:
    """Fingerprint model contents to prevent reuse of stale tables."""
    return hashlib.sha256(Path(model.model_path).read_bytes()).hexdigest()[:12]


if TYPE_CHECKING:
    from .fmm.grid import FmmTravelTimeGrid1D
    from .tau.grid import TauTravelTimeGrid

__all__ = [
    "LayeredGrid1D",
    "FmmTravelTimeGrid1D",
    "TauTravelTimeGrid",
]


@njit(cache=True, parallel=True)
def numerical_derivatives_1d_tables_batch(
    depths: np.ndarray,
    distances: np.ndarray,
    azimuths_deg: np.ndarray,
    delta: float,
    depth_grid: np.ndarray,
    distance_grid: np.ndarray,
    table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = distances.size
    travel_times = np.empty(n, dtype=np.float64)
    dt_dr = np.empty(n, dtype=np.float64)
    dt_dx = np.empty(n, dtype=np.float64)
    dt_dy = np.empty(n, dtype=np.float64)
    dt_dz = np.empty(n, dtype=np.float64)

    min_depth = depth_grid[0]
    max_depth = depth_grid[-1]
    min_distance = distance_grid[0]
    max_distance = distance_grid[-1]
    two_delta = 2.0 * delta

    for i in prange(n):
        depth = depths[i]
        distance = distances[i]
        travel_time = interp_quant(depth, distance, depth_grid, distance_grid, table)
        travel_times[i] = travel_time

        if distance - delta < min_distance:
            tt_plus = interp_quant(
                depth, distance + delta, depth_grid, distance_grid, table
            )
            dt_ddistance = (tt_plus - travel_time) / delta
        elif distance + delta > max_distance:
            tt_minus = interp_quant(
                depth, distance - delta, depth_grid, distance_grid, table
            )
            dt_ddistance = (travel_time - tt_minus) / delta
        else:
            tt_plus = interp_quant(
                depth, distance + delta, depth_grid, distance_grid, table
            )
            tt_minus = interp_quant(
                depth, distance - delta, depth_grid, distance_grid, table
            )
            dt_ddistance = (tt_plus - tt_minus) / two_delta

        if depth - delta < min_depth:
            tt_plus = interp_quant(
                depth + delta, distance, depth_grid, distance_grid, table
            )
            dt_ddepth = (tt_plus - travel_time) / delta
        elif depth + delta > max_depth:
            tt_minus = interp_quant(
                depth - delta, distance, depth_grid, distance_grid, table
            )
            dt_ddepth = (travel_time - tt_minus) / delta
        else:
            tt_plus = interp_quant(
                depth + delta, distance, depth_grid, distance_grid, table
            )
            tt_minus = interp_quant(
                depth - delta, distance, depth_grid, distance_grid, table
            )
            dt_ddepth = (tt_plus - tt_minus) / two_delta

        radial = -dt_ddistance
        azimuth = np.deg2rad(azimuths_deg[i])
        dt_dr[i] = radial
        dt_dx[i] = radial * np.sin(azimuth)
        dt_dy[i] = radial * np.cos(azimuth)
        dt_dz[i] = dt_ddepth

    return travel_times, dt_dr, dt_dx, dt_dy, dt_dz


class LayeredGrid1D(LayeredPropagationEngine):
    """Shared 1-D travel-time table backend.

    Subclasses build or load ``depth_grid``, ``distance_grid``,
    ``travel_time_tables``, and ``takeoff_angle_tables``. Query,
    interpolation, and batched derivatives live here.
    """

    feature_flags = frozenset(
        {
            "propagation.layered",
            "propagation.table",
            "derivative.batch",
            "derivative.numerical",
        }
    )

    def __init__(self, model: VelocityModel, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.depth_grid: np.ndarray
        self.distance_grid: np.ndarray
        self.travel_time_tables: dict[str, np.ndarray] = {}
        self.takeoff_angle_tables: dict[str, np.ndarray] = {}

    def save(self, filename: str | Path) -> None:
        """Save grids and all phase tables to a compressed ``.npz`` file."""
        if not self.travel_time_tables or not self.takeoff_angle_tables:
            raise ValueError("Travel-time tables are not built; nothing to save.")
        tables = {f"tt_{ph}": t for ph, t in self.travel_time_tables.items()}
        tables |= {f"ta_{ph}": t for ph, t in self.takeoff_angle_tables.items()}
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            filename,
            depth_grid=self.depth_grid,
            distance_grid=self.distance_grid,
            **tables,
        )

    def load(self, filename: str | Path, phase: str) -> None:
        """Load grids and the tables for ``phase``; ``ValueError`` if absent."""
        with np.load(filename) as data:
            for key in (f"tt_{phase}", f"ta_{phase}"):
                if key not in data:
                    raise ValueError(f"Phase {phase} not found in {filename}.")
            self.depth_grid = data["depth_grid"]
            self.distance_grid = data["distance_grid"]
            self.travel_time_tables[phase] = data[f"tt_{phase}"]
            self.takeoff_angle_tables[phase] = data[f"ta_{phase}"]

    @property
    def name(self) -> str:
        """Return a human-readable name for this travel-time grid."""
        return f"LayeredGrid1D({self.velocity_model.model_name})"

    def _resolve_phase_table(self, phase: str) -> str:
        """Return the stored phase key matching ``phase``.

        Raises ``ValueError`` when the phase is not tabulated.
        """
        wanted = normalize_phase(phase)
        for stored in self.travel_time_tables:
            if normalize_phase(stored) == wanted:
                return stored
        raise ValueError(f"Phase {phase!r} not in available phases: {self.phases}")

    def _depth_grid_array(self) -> np.ndarray:
        return np.asarray(self.depth_grid, dtype=np.float64)

    def _distance_grid_array(self) -> np.ndarray:
        return np.asarray(self.distance_grid, dtype=np.float64)

    def _travel_time_table(self, phase: str) -> np.ndarray:
        key = self._resolve_phase_table(phase)
        return np.asarray(self.travel_time_tables[key], dtype=np.float64)

    def _takeoff_angle_table(self, phase: str) -> np.ndarray:
        key = self._resolve_phase_table(phase)
        return np.asarray(self.takeoff_angle_tables[key], dtype=np.float64)

    def _effective_max_distance_km(self, max_distance_km: float | None) -> float:
        table_max = float(self.distance_grid[-1])
        configured = (
            max_distance_km if max_distance_km is not None else self.max_distance_km
        )
        if configured is None:
            configured = getattr(self, "max_distance", None)
        if configured is None:
            return table_max
        return min(float(configured), table_max)

    def _effective_max_depth_km(self) -> float:
        return float(self.depth_grid[-1])

    def _planet_radius_m(self) -> float:
        return float(self.velocity_model.radius_of_planet_km) * 1000.0

    def _geometry_batch(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: np.ndarray,
        receiver_longitude: np.ndarray,
        receiver_elevation_m: np.ndarray,
        max_distance_km: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        receiver_latitude = np.asarray(receiver_latitude, dtype=np.float64)
        receiver_longitude = np.asarray(receiver_longitude, dtype=np.float64)
        receiver_elevation_m = np.asarray(receiver_elevation_m, dtype=np.float64)

        distances_m, azimuths_deg, _ = gps2dist_azimuth_batch_nb(
            float(source_latitude),
            float(source_longitude),
            receiver_latitude,
            receiver_longitude,
            self._planet_radius_m(),
        )
        distances_km = distances_m / 1000.0
        depths_km = self.effective_source_depth_km(
            source_depth_km, receiver_elevation_m
        )
        max_distance = self._effective_max_distance_km(max_distance_km)
        in_range = (distances_km <= max_distance) & (
            depths_km <= self._effective_max_depth_km()
        )
        return distances_km, azimuths_deg, depths_km, in_range

    def compute_travel_time(
        self, distance_km: float, depth_km: float, phase: str
    ) -> float:
        """Interpolate a tabulated travel time for one 1-D query."""
        return float(
            interp_quant(
                depth_km,
                distance_km,
                self._depth_grid_array(),
                self._distance_grid_array(),
                self._travel_time_table(phase),
            )
        )

    def compute_travel_time_batch(
        self,
        distance_km: np.ndarray,
        depth_km: np.ndarray,
        phase: str,
    ) -> np.ndarray:
        """Vectorized :meth:`compute_travel_time`."""
        return interp_quant_batch(
            np.asarray(depth_km, dtype=np.float64),
            np.asarray(distance_km, dtype=np.float64),
            self._depth_grid_array(),
            self._distance_grid_array(),
            self._travel_time_table(phase),
        )

    def compute_takeoff_angle(
        self, distance_km: float, depth_km: float, phase: str
    ) -> float:
        """Interpolate a tabulated takeoff angle for one 1-D query."""
        return float(
            interp_quant(
                depth_km,
                distance_km,
                self._depth_grid_array(),
                self._distance_grid_array(),
                self._takeoff_angle_table(phase),
            )
        )

    def compute_takeoff_angle_batch(
        self,
        distance_km: np.ndarray,
        depth_km: np.ndarray,
        phase: str,
    ) -> np.ndarray:
        """Vectorized :meth:`compute_takeoff_angle`."""
        return interp_quant_batch(
            np.asarray(depth_km, dtype=np.float64),
            np.asarray(distance_km, dtype=np.float64),
            self._depth_grid_array(),
            self._distance_grid_array(),
            self._takeoff_angle_table(phase),
        )

    def get_arrival_1d(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ) -> tuple[float, float]:
        """Return travel time and takeoff angle for a 1-D table query."""
        self._resolve_phase_table(phase)
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
        return float(time), float(takeoff_angle)

    def get_arrival_1d_batch(
        self,
        arc_distance_km: np.ndarray,
        source_depth_km: np.ndarray,
        phase: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized :meth:`get_arrival_1d`. Invalid samples are ``NaN``."""
        self._resolve_phase_table(phase)
        distances = np.abs(np.asarray(arc_distance_km, dtype=np.float64))
        depths = np.asarray(source_depth_km, dtype=np.float64)
        return (
            self.compute_travel_time_batch(distances, depths, phase),
            self.compute_takeoff_angle_batch(distances, depths, phase),
        )

    def _limits(
        self, max_distance_km: float | None
    ) -> tuple[float | None, float | None]:
        """Clamp ``get_arrival`` coverage to the tabulated grid."""
        return (
            self._effective_max_distance_km(max_distance_km),
            self._effective_max_depth_km(),
        )

    def get_arrival_batch(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: np.ndarray,
        receiver_longitude: np.ndarray,
        receiver_elevation_m: np.ndarray,
        phase: str,
        max_distance_km: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized :meth:`get_arrival`.

        Returns ``(distances, travel_times, takeoff_angles, azimuths)``.
        Out-of-range travel times and takeoff angles are ``NaN``.
        """
        self._resolve_phase_table(phase)
        distances, azimuths, depths, in_range = self._geometry_batch(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            max_distance_km=max_distance_km,
        )
        travel_times = np.full(distances.shape, np.nan, dtype=np.float64)
        takeoff_angles = np.full(distances.shape, np.nan, dtype=np.float64)
        if in_range.any():
            times, angles = self.get_arrival_1d_batch(
                distances[in_range], depths[in_range], phase
            )
            travel_times[in_range] = times
            takeoff_angles[in_range] = angles
        return distances, travel_times, takeoff_angles, azimuths

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
        """Compute scalar slowness derivatives, gated by feature flags."""
        if mode == "analytical" and "derivative.analytical" not in self.feature_flags:
            raise NotImplementedError(
                "Analytical derivatives are not implemented for this travel-time grid."
            )
        return super().compute_derivatives(
            arc_distance_km=arc_distance_km,
            source_depth_km=source_depth_km,
            az_deg=az_deg,
            phase=phase,
            takeoff_angle_deg=takeoff_angle_deg,
            delta=delta,
            mode=mode,
        )

    def compute_derivatives_batch(
        self,
        arc_distance_km: np.ndarray,
        source_depth_km: np.ndarray,
        az_deg: np.ndarray,
        phase: str,
        takeoff_angle_deg: np.ndarray | None,
        delta: Optional[float] = 0.01,
        mode: str = "analytical",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized :meth:`compute_derivatives`."""
        distances = np.asarray(arc_distance_km, dtype=np.float64)
        depths = np.asarray(source_depth_km, dtype=np.float64)
        azimuths = np.asarray(az_deg, dtype=np.float64)

        if mode == "analytical":
            if "derivative.analytical" not in self.feature_flags:
                raise NotImplementedError(
                    "Analytical derivatives are not implemented for this "
                    "travel-time grid."
                )
            if takeoff_angle_deg is None:
                takeoff_angle_deg = self.compute_takeoff_angle_batch(
                    distances, depths, phase
                )
            takeoffs = np.asarray(takeoff_angle_deg, dtype=np.float64)
            velocities = np.fromiter(
                (
                    self.velocity_model.get_velocity_1d(float(depth), phase)
                    for depth in depths
                ),
                dtype=np.float64,
                count=depths.size,
            )
            dt_dx, dt_dy, dt_dz = get_analytical_derivative_1d_batch(
                takeoffs, azimuths, velocities
            )
            dt_dr = -np.sin(np.deg2rad(takeoffs)) / velocities
            return dt_dr, dt_dx, dt_dy, dt_dz

        if mode == "numerical":
            if delta is None:
                raise ValueError(
                    "Delta must be provided for numerical derivative computation."
                )
            _travel_times, dt_dr, dt_dx, dt_dy, dt_dz = (
                numerical_derivatives_1d_tables_batch(
                    depths,
                    distances,
                    azimuths,
                    float(delta),
                    self._depth_grid_array(),
                    self._distance_grid_array(),
                    self._travel_time_table(phase),
                )
            )
            return dt_dr, dt_dx, dt_dy, dt_dz

        raise ValueError(
            f"Invalid mode: {mode}. Supported modes are analytical and numerical."
        )

    def compute_travel_time_params_batch(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: np.ndarray,
        receiver_longitude: np.ndarray,
        receiver_elevation_m: np.ndarray,
        phase: str,
        delta: float | None = None,
        derivative_method: str = "analytical",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized :meth:`compute_travel_time_params`.

        Returns ``(distances, travel_times, dt_dr, dt_dx, dt_dy, dt_dz)``.
        Out-of-range or failed samples are ``NaN``.
        """
        self._resolve_phase_table(phase)
        distances, azimuths, depths, in_range = self._geometry_batch(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            max_distance_km=self.max_distance_km,
        )
        travel_times = np.full(distances.shape, np.nan, dtype=np.float64)
        dt_dr = np.full(distances.shape, np.nan, dtype=np.float64)
        dt_dx = np.full(distances.shape, np.nan, dtype=np.float64)
        dt_dy = np.full(distances.shape, np.nan, dtype=np.float64)
        dt_dz = np.full(distances.shape, np.nan, dtype=np.float64)
        if not in_range.any():
            return distances, travel_times, dt_dr, dt_dx, dt_dy, dt_dz

        active_distances = distances[in_range]
        active_depths = depths[in_range]
        active_azimuths = azimuths[in_range]

        if derivative_method == "numerical":
            if delta is None:
                raise ValueError(
                    "Delta must be provided for numerical derivative computation."
                )
            times, radial, dx, dy, dz = numerical_derivatives_1d_tables_batch(
                active_depths,
                active_distances,
                active_azimuths,
                float(delta),
                self._depth_grid_array(),
                self._distance_grid_array(),
                self._travel_time_table(phase),
            )
        else:
            times = self.compute_travel_time_batch(
                active_distances, active_depths, phase
            )
            takeoffs = self.compute_takeoff_angle_batch(
                active_distances, active_depths, phase
            )
            radial, dx, dy, dz = self.compute_derivatives_batch(
                arc_distance_km=active_distances,
                source_depth_km=active_depths,
                az_deg=active_azimuths,
                phase=phase,
                takeoff_angle_deg=takeoffs,
                delta=delta,
                mode=derivative_method,
            )

        travel_times[in_range] = times
        dt_dr[in_range] = radial
        dt_dx[in_range] = dx
        dt_dy[in_range] = dy
        dt_dz[in_range] = dz
        return distances, travel_times, dt_dr, dt_dx, dt_dy, dt_dz


def __getattr__(name: str):
    """Lazy load grid backends to avoid unnecessary dependencies."""
    if name == "FmmTravelTimeGrid1D":
        from .fmm.grid import FmmTravelTimeGrid1D

        return FmmTravelTimeGrid1D
    elif name == "TauTravelTimeGrid":
        from .tau.grid import TauTravelTimeGrid

        return TauTravelTimeGrid
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
