"""Volumetric travel-time grid backend using pyekfmm on a 3D velocity grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pyekfmm as fmm  # type: ignore[import-untyped]

from hypy.logger import logger
from hypy.model import VelocityModel
from hypy.model.volumetric import GridSliceData, VelocityGridProvider
from hypy.propagation.volumetric import VolumetricPropagationEngine
from hypy.utils.seismic import normalize_phase


@dataclass(frozen=True)
class _ReceiverTable:
    """Per-receiver cached eikonal solution on a local sampled grid."""

    tt_table: np.ndarray
    grid: GridSliceData
    ax: list[float]
    ay: list[float]
    az: list[float]
    nx: int
    ny: int
    nz: int
    is_complete: bool


class FmmTravelTimeGrid3D(VolumetricPropagationEngine):
    """Travel-time table backend using pyekfmm on a 3D velocity grid."""

    feature_flags = frozenset(
        {
            "propagation.volumetric",
            "propagation.table",
            "derivative.numerical",
        }
    )
    required_model_features = frozenset({"model.grid.3d"})

    tt_table: Optional[np.ndarray] = None
    tt_tables_by_receiver: dict[tuple[str, tuple[float, float, float]], _ReceiverTable]

    def __init__(
        self,
        model: Union[VelocityModel, str, Path],
        phases: Sequence[str],
        max_depth_km: float,
        max_distance_km: float,
        depth_step_km: float,
        distance_step_km: float,
        order: int = 2,
    ):
        if not phases:
            raise ValueError("At least one phase must be provided.")

        normalized_phases = tuple(normalize_phase(phase) for phase in phases)
        super().__init__(
            model,
            mode="table",
            engine="pyekfmm",
            max_depth_km=max_depth_km,
            max_distance_km=max_distance_km,
            depth_step_km=depth_step_km,
            distance_step_km=distance_step_km,
            phases=normalized_phases,
        )

        if not callable(getattr(self.velocity_model, "get_grid", None)):
            raise TypeError(
                f"Volumetric model {type(self.velocity_model).__name__!r} "
                "does not implement get_grid()."
            )
        # NCGrid / NllGridModel both implement the grid-provider protocol.
        self.grid_provider: VelocityGridProvider = self.velocity_model

        max_distance_km = float(self.max_distance_km)
        distance_step_km = float(self.distance_step_km)
        max_depth_km = float(self.max_depth_km)
        depth_step_km = float(self.depth_step_km)
        self.local_nx = int(round(2.0 * max_distance_km / distance_step_km)) + 1
        self.local_ny = int(round(2.0 * max_distance_km / distance_step_km)) + 1
        self.local_nz = int(round(max_depth_km / depth_step_km)) + 1
        self.order = int(order)

        self._grid_cache: dict[tuple[str, float, float, float], GridSliceData] = {}
        self._cached_receiver_phase_xyz: Optional[
            tuple[str, tuple[float, float, float]]
        ] = None
        self.tt_tables_by_receiver = {}

    @staticmethod
    def _normalize_phase(phase: str) -> str:
        return normalize_phase(phase)

    @staticmethod
    def _project_to_xy(projected: object) -> tuple[float, float]:
        arr = np.asarray(projected, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            raise ValueError("Projected coordinates must contain at least x and y.")
        return float(arr[0]), float(arr[1])

    @staticmethod
    def _round_coord(value: float) -> float:
        return float(round(float(value), 5))

    def _get_receiver_grid(
        self,
        phase: str,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_depth_km: float,
    ) -> GridSliceData:
        phase_code = self._normalize_phase(phase)
        key = (
            phase_code,
            self._round_coord(receiver_latitude),
            self._round_coord(receiver_longitude),
            self._round_coord(receiver_depth_km),
        )
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached

        grid = self.grid_provider.get_grid(
            center_latitude=receiver_latitude,
            center_longitude=receiver_longitude,
            center_depth_km=receiver_depth_km,
            max_depth_km=self.max_depth_km,
            max_distance_km=self.max_distance_km,
            grid_latitude_spacing_km=self.distance_step_km,
            grid_longitude_spacing_km=self.distance_step_km,
            grid_depth_spacing_km=self.depth_step_km,
            phase=phase_code,
        )
        if grid.array.size == 0:
            raise ValueError(f"Velocity grid is empty for phase {phase_code}.")

        self._grid_cache[key] = grid
        return grid

    def _compute_tt_table(
        self,
        phase: str,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_depth_km: float,
    ) -> _ReceiverTable:
        phase_code = self._normalize_phase(phase)
        receiver_grid = self._get_receiver_grid(
            phase=phase_code,
            receiver_latitude=receiver_latitude,
            receiver_longitude=receiver_longitude,
            receiver_depth_km=receiver_depth_km,
        )

        receiver_x, receiver_y = self._project_to_xy(
            receiver_grid.project(receiver_latitude, receiver_longitude)
        )
        receiver_xyz = (receiver_x, receiver_y, float(receiver_depth_km))
        cache_key = (
            phase_code,
            self._round_coord(receiver_latitude),
            self._round_coord(receiver_longitude),
            self._round_coord(receiver_depth_km),
        )

        cached_result = self.tt_tables_by_receiver.get(cache_key)
        if cached_result is not None:
            self.tt_table = cached_result.tt_table
            self._cached_receiver_phase_xyz = cache_key
            return cached_result

        if not receiver_grid.contains_xyz(receiver_x, receiver_y, receiver_depth_km):
            result = _ReceiverTable(
                tt_table=np.empty((0, 0, 0), dtype=np.float64),
                grid=receiver_grid,
                ax=[receiver_grid.x_orig, receiver_grid.dx, receiver_grid.nx],
                ay=[receiver_grid.y_orig, receiver_grid.dy, receiver_grid.ny],
                az=[receiver_grid.z_orig, receiver_grid.dz, receiver_grid.nz],
                nx=receiver_grid.nx,
                ny=receiver_grid.ny,
                nz=receiver_grid.nz,
                is_complete=False,
            )
            self.tt_tables_by_receiver[cache_key] = result
            return result

        sampled_velocity = np.asarray(receiver_grid.array, dtype=np.float32)
        ax = [
            float(receiver_grid.x_orig),
            float(receiver_grid.dx),
            int(receiver_grid.nx),
        ]
        ay = [
            float(receiver_grid.y_orig),
            float(receiver_grid.dy),
            int(receiver_grid.ny),
        ]
        az = [
            float(receiver_grid.z_orig),
            float(receiver_grid.dz),
            int(receiver_grid.nz),
        ]

        times_flat = fmm.eikonal(
            sampled_velocity.flatten(order="F"),
            xyz=np.array(receiver_xyz, dtype=np.float32),
            ax=ax,
            ay=ay,
            az=az,
            order=self.order,
            verb=0,
        )

        tt_table = np.asarray(times_flat, dtype=np.float64).reshape(
            receiver_grid.nx, receiver_grid.ny, receiver_grid.nz, order="F"
        )
        result = _ReceiverTable(
            tt_table=tt_table,
            grid=receiver_grid,
            ax=ax,
            ay=ay,
            az=az,
            nx=receiver_grid.nx,
            ny=receiver_grid.ny,
            nz=receiver_grid.nz,
            is_complete=True,
        )

        self.tt_tables_by_receiver[cache_key] = result
        self.tt_table = tt_table
        self._cached_receiver_phase_xyz = cache_key
        return result

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
    ) -> tuple[float, float | None, float | None, float]:
        """Return arrival parameters for a given source-receiver configuration.

        Returns the arc distance, travel time,
        takeoff angle (currently None), and azimuth.

        Parameters
        ----------
        source_latitude : float
            Latitude of the seismic source in degrees.
        source_longitude : float
            Longitude of the seismic source in degrees.
        source_depth_km : float
            Depth of the seismic source in kilometers.
        receiver_latitude : float
            Latitude of the receiver in degrees.
        receiver_longitude : float
            Longitude of the receiver in degrees.
        receiver_elevation_m : float
            Elevation of the receiver in meters.
        phase : str
            Seismic phase for which to compute arrivals.
        max_distance_km : float | None, optional
            Maximum distance for which to compute arrivals, by default None

        Returns
        -------
        tuple[float, float | None, float | None, float]
            Arc distance, travel time, takeoff angle, and azimuth source-receiver.
        """
        arc_distance_km, azimuth_deg, _ = self._compute_arc_params(
            source_latitude,
            source_longitude,
            receiver_latitude,
            receiver_longitude,
        )
        if max_distance_km is not None and arc_distance_km > float(max_distance_km):
            logger.info(
                "Station at distance %.2f km exceeds max_distance %.2f km. Skipping.",
                arc_distance_km,
                max_distance_km,
            )
            return arc_distance_km, None, None, azimuth_deg

        phase_code = self._normalize_phase(phase)
        eps = 1e-3  # km = 1 m
        receiver_z = max(eps, -float(receiver_elevation_m) / 1000.0)
        source_z = float(source_depth_km)

        try:
            receiver_table = self._compute_tt_table(
                phase=phase_code,
                receiver_latitude=receiver_latitude,
                receiver_longitude=receiver_longitude,
                receiver_depth_km=receiver_z,
            )
        except ValueError:
            return arc_distance_km, None, None, azimuth_deg

        grid = receiver_table.grid

        receiver_x, receiver_y = self._project_to_xy(
            grid.project(receiver_latitude, receiver_longitude)
        )
        source_x, source_y = self._project_to_xy(
            grid.project(source_latitude, source_longitude)
        )

        if not grid.contains_xyz(receiver_x, receiver_y, receiver_z):
            return arc_distance_km, None, None, azimuth_deg

        if not grid.contains_xyz(source_x, source_y, source_z):
            return arc_distance_km, None, None, azimuth_deg

        if receiver_table.tt_table.size == 0:
            logger.warning(
                "(%s, %s, %s km) TT table is empty. Returning no arrival.",
                receiver_latitude,
                receiver_longitude,
                receiver_z,
            )
            return arc_distance_km, None, None, azimuth_deg

        fx = (source_x - receiver_table.ax[0]) / receiver_table.ax[1]
        fy = (source_y - receiver_table.ay[0]) / receiver_table.ay[1]
        fz = (source_z - receiver_table.az[0]) / receiver_table.az[1]

        if (
            0.0 <= fx <= receiver_table.nx - 1
            and 0.0 <= fy <= receiver_table.ny - 1
            and 0.0 <= fz <= receiver_table.nz - 1
        ):
            ix0 = int(np.floor(fx))
            iy0 = int(np.floor(fy))
            iz0 = int(np.floor(fz))
            ix1 = min(ix0 + 1, receiver_table.nx - 1)
            iy1 = min(iy0 + 1, receiver_table.ny - 1)
            iz1 = min(iz0 + 1, receiver_table.nz - 1)

            tx = float(fx - ix0)
            ty = float(fy - iy0)
            tz = float(fz - iz0)

            c000 = receiver_table.tt_table[ix0, iy0, iz0]
            c100 = receiver_table.tt_table[ix1, iy0, iz0]
            c010 = receiver_table.tt_table[ix0, iy1, iz0]
            c110 = receiver_table.tt_table[ix1, iy1, iz0]
            c001 = receiver_table.tt_table[ix0, iy0, iz1]
            c101 = receiver_table.tt_table[ix1, iy0, iz1]
            c011 = receiver_table.tt_table[ix0, iy1, iz1]
            c111 = receiver_table.tt_table[ix1, iy1, iz1]

            corners = np.array([c000, c100, c010, c110, c001, c101, c011, c111])
            if np.all(np.isfinite(corners)):
                c00 = c000 * (1.0 - tx) + c100 * tx
                c10 = c010 * (1.0 - tx) + c110 * tx
                c01 = c001 * (1.0 - tx) + c101 * tx
                c11 = c011 * (1.0 - tx) + c111 * tx
                c0 = c00 * (1.0 - ty) + c10 * ty
                c1 = c01 * (1.0 - ty) + c11 * ty
                travel_time = float(c0 * (1.0 - tz) + c1 * tz)
            else:
                travel_time = None
        else:
            travel_time = None

        return arc_distance_km, travel_time, None, azimuth_deg
