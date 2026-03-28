from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
from nllgrid import NLLGrid  # type: ignore[import-untyped]

from hypy.propagation.volumetric import VolumetricPropagationModel
from hypy.utils.nllgrid import create_nll_grid_from_tvel

import pyekfmm as fmm  # type: ignore[import-untyped]


@dataclass(frozen=True)
class _ReceiverTable:
    """Per-receiver cached eikonal solution on a local sampled grid."""

    tt_table: np.ndarray
    ax: list[float]
    ay: list[float]
    az: list[float]
    nx: int
    ny: int
    nz: int


class PyekfmmTimeTables(VolumetricPropagationModel):
    """Ray tracer backend using pyekfmm on an NLL velocity grid."""

    nll_grid: NLLGrid
    tt_table: Optional[np.ndarray] = None
    tt_tables_by_receiver: dict[tuple[str, tuple[float, float, float]], _ReceiverTable]

    def __init__(
        self,
        model_path: str,
        max_distance_km: float,
        distance_step_km: float,
        max_depth_km: float,
        depth_step_km: float,
        phases: list[str],
        order: int = 2,
    ):
        if not phases:
            raise ValueError("At least one phase must be provided.")

        self.phases = tuple(sorted({self._normalize_phase(phase) for phase in phases}))
        self.nll_grids_by_phase: dict[str, NLLGrid] = {}
        self.velocities_by_phase: dict[str, np.ndarray] = {}

        self.max_distance_km = float(max_distance_km)
        self.distance_step_km = float(distance_step_km)
        self.max_depth_km = float(max_depth_km)
        self.depth_step_km = float(depth_step_km)
        self.local_nx = int(2 * self.max_distance_km / self.distance_step_km) + 1
        self.local_ny = int(2 * self.max_distance_km / self.distance_step_km) + 1
        self.local_nz = int(self.max_depth_km / self.depth_step_km) + 1

        if Path(model_path).suffix == ".tvel":
            for phase in self.phases:
                nll_grid = create_nll_grid_from_tvel(
                    model_path,
                    phase=phase,
                    nx=int(20 * max_distance_km / distance_step_km) + 1,
                    ny=int(20 * max_distance_km / distance_step_km) + 1,
                    nz=int(max_depth_km / depth_step_km) + 1,
                    x_orig=-max_distance_km * 10,
                    y_orig=-max_distance_km * 10,
                    z_orig=0.0,
                    dx=distance_step_km,
                    dy=distance_step_km,
                    dz=depth_step_km,
                    orig_lat=10.0,
                    orig_lon=10.0,
                    proj_name="SIMPLE",
                )

                if nll_grid.array is None:
                    raise ValueError(f"NLLGrid array is None for phase {phase}.")

                velocity = np.asarray(nll_grid.array, dtype=np.float32)
                self.nll_grids_by_phase[phase] = nll_grid
                self.velocities_by_phase[phase] = velocity
        else:
            nll_grid = NLLGrid(model_path)
            if nll_grid.array is None:
                raise ValueError("NLLGrid array is None")
            velocity = np.asarray(nll_grid.array, dtype=np.float32)

            for phase in self.phases:
                self.nll_grids_by_phase[phase] = nll_grid
                self.velocities_by_phase[phase] = velocity

        primary_phase = self.phases[0]
        self.nll_grid = self.nll_grids_by_phase[primary_phase]
        self.order = order
        self.velocity = self.velocities_by_phase[primary_phase]

        self._cached_receiver_phase_xyz: Optional[
            tuple[str, tuple[float, float, float]]
        ] = None
        self.tt_tables_by_receiver = {}

    @staticmethod
    def _normalize_phase(phase: str) -> str:
        phase_code = phase.strip().upper()
        if phase_code not in {"P", "S"}:
            raise ValueError(
                f"Unsupported phase: {phase!r}. Supported phases are 'P' and 'S'."
            )
        return phase_code

    @staticmethod
    def _project_to_xy(projected: object) -> tuple[float, float]:
        arr = np.asarray(projected, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            raise ValueError("Projected coordinates must contain at least x and y.")
        return float(arr[0]), float(arr[1])

    def _compute_tt_table(
        self,
        phase: str,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_depth_km: float,
    ) -> _ReceiverTable:
        phase_code = self._normalize_phase(phase)
        nll_grid = self.nll_grids_by_phase[phase_code]

        receiver_x, receiver_y = self._project_to_xy(
            nll_grid.project(receiver_longitude, receiver_latitude)
        )
        receiver_xyz = (receiver_x, receiver_y, float(receiver_depth_km))
        cache_key = (phase_code, receiver_xyz)

        cached_result = self.tt_tables_by_receiver.get(cache_key)
        if cached_result is not None:
            self.tt_table = cached_result.tt_table
            self._cached_receiver_phase_xyz = cache_key
            return cached_result

        ax = [receiver_x - self.max_distance_km, self.distance_step_km, self.local_nx]
        ay = [receiver_y - self.max_distance_km, self.distance_step_km, self.local_ny]
        az = [float(nll_grid.z_orig), self.depth_step_km, self.local_nz]

        sampled_velocity = self._sample_velocity_grid(
            phase=phase_code,
            ax=ax,
            ay=ay,
            az=az,
        )

        times_flat = fmm.eikonal(  # type: ignore[union-attr]
            sampled_velocity.flatten(order="F"),
            xyz=np.array(receiver_xyz, dtype=np.float32),
            ax=ax,
            ay=ay,
            az=az,
            order=self.order,
            verb=0,
        )

        tt_table = np.asarray(times_flat, dtype=np.float64).reshape(
            self.local_nx, self.local_ny, self.local_nz, order="F"
        )
        result = _ReceiverTable(
            tt_table=tt_table,
            ax=ax,
            ay=ay,
            az=az,
            nx=self.local_nx,
            ny=self.local_ny,
            nz=self.local_nz,
        )

        self.tt_tables_by_receiver[cache_key] = result
        self.tt_table = tt_table
        self._cached_receiver_phase_xyz = cache_key
        return result

    def _sample_velocity_grid(
        self,
        phase: str,
        ax: list[float],
        ay: list[float],
        az: list[float],
    ) -> np.ndarray:
        """Sample a receiver-centered local cube from the phase NLL velocity grid."""

        velocity = self.velocities_by_phase[phase]
        nll_grid = self.nll_grids_by_phase[phase]
        nx_src, ny_src, nz_src = velocity.shape

        x_coords = ax[0] + np.arange(int(ax[2]), dtype=np.float64) * ax[1]
        y_coords = ay[0] + np.arange(int(ay[2]), dtype=np.float64) * ay[1]
        z_coords = az[0] + np.arange(int(az[2]), dtype=np.float64) * az[1]

        ix = np.rint((x_coords - float(nll_grid.x_orig)) / float(nll_grid.dx)).astype(
            int
        )
        iy = np.rint((y_coords - float(nll_grid.y_orig)) / float(nll_grid.dy)).astype(
            int
        )
        iz = np.rint((z_coords - float(nll_grid.z_orig)) / float(nll_grid.dz)).astype(
            int
        )

        ix = np.clip(ix, 0, nx_src - 1)
        iy = np.clip(iy, 0, ny_src - 1)
        iz = np.clip(iz, 0, nz_src - 1)

        return np.asarray(velocity[np.ix_(ix, iy, iz)], dtype=np.float32)

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
        phase_code = self._normalize_phase(phase)
        nll_grid = self.nll_grids_by_phase[phase_code]

        receiver_x, receiver_y = self._project_to_xy(
            nll_grid.project(receiver_longitude, receiver_latitude)
        )
        source_x, source_y = self._project_to_xy(
            nll_grid.project(source_longitude, source_latitude)
        )
        source_z = float(source_depth_km)
        eps = 1e-3  # km = 1 m
        receiver_z = max(eps, -float(receiver_elevation_m) / 1000.0)

        receiver_table = self._compute_tt_table(
            phase=phase_code,
            receiver_latitude=receiver_latitude,
            receiver_longitude=receiver_longitude,
            receiver_depth_km=receiver_z,
        )

        fx = (source_x - receiver_table.ax[0]) / receiver_table.ax[1]
        fy = (source_y - receiver_table.ay[0]) / receiver_table.ay[1]
        fz = (source_z - receiver_table.az[0]) / receiver_table.az[1]

        # Trilinear interpolation for smooth travel-time evaluation.
        # This avoids piecewise-constant nearest-neighbor sampling, which can
        # collapse finite-difference derivatives to zero.
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

        arc_distance_km = float(
            np.sqrt(
                (receiver_x - source_x) ** 2
                + (receiver_y - source_y) ** 2
                + (receiver_z - source_z) ** 2
            )
        )

        # Keep output compatible with existing tracer API.
        return arc_distance_km, travel_time, None, 0.0
