"""NLLGrid-backed 3D velocity grid loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from nllgrid import NLLGrid  # type: ignore[import-untyped]

from hypy.model import TvelModel, VelocityModel
from hypy.utils.nllgrid import create_nll_grid_from_tvel
from hypy.utils.seismic import normalize_phase

from .. import GridSliceData


class NllGridModel(VelocityModel):
    """3D velocity grid support using NLLGrid or TVEL -> NLLGrid conversion."""

    feature_flags = frozenset(
        {
            "model.volumetric",
            "model.grid.3d",
            "format.nllgrid",
            "phase.p",
            "phase.s",
        }
    )

    def __init__(
        self,
        model_path: Path | str,
        radius_of_planet_km: float = 6371.0,
    ):
        model_path = Path(model_path).expanduser().resolve()
        super().__init__(
            model_path=model_path,
            model_name=model_path.stem,
            dim=3,
            radius_of_planet_km=radius_of_planet_km,
        )

        self.model_path = model_path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Velocity model file not found: {self.model_path}")

        self.is_tvel = self.model_path.suffix.lower() == ".tvel"
        self._tvel_model: TvelModel | None = None
        self._nll_grid: NLLGrid | None = None
        self._phase_grids: dict[str, NLLGrid] = {}
        self._wrapped_phase_grids: dict[tuple[str, str], GridSliceData] = {}

        if not self.is_tvel:
            nll_grid = NLLGrid(self.model_path.as_posix())
            if getattr(nll_grid, "array", None) is None:
                raise ValueError("NLLGrid array is None")
            self._nll_grid = nll_grid

    @staticmethod
    def _normalize_phase(phase: str) -> str:
        return normalize_phase(phase).upper()

    @staticmethod
    def _normalize_grid_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "nx",
            "ny",
            "nz",
            "dx",
            "dy",
            "dz",
            "x_orig",
            "y_orig",
            "z_orig",
            "orig_lat",
            "orig_lon",
            "proj_name",
            "first_std_paral",
            "second_std_paral",
            "proj_ellipsoid",
        }
        return {k: v for k, v in kwargs.items() if k in allowed}

    @staticmethod
    def _wrap_nll_grid(
        grid: NLLGrid,
        requested_nx: int | None = None,
        requested_ny: int | None = None,
        requested_nz: int | None = None,
    ) -> GridSliceData:
        array = np.asarray(grid.array, dtype=np.float32)
        nx, ny, nz = array.shape

        return GridSliceData(
            array=array,
            orig_lat=float(getattr(grid, "orig_lat", 0.0)),
            orig_lon=float(getattr(grid, "orig_lon", 0.0)),
            x_orig=float(getattr(grid, "x_orig", 0.0)),
            y_orig=float(getattr(grid, "y_orig", 0.0)),
            z_orig=float(getattr(grid, "z_orig", 0.0)),
            dx=float(getattr(grid, "dx", 1.0)),
            dy=float(getattr(grid, "dy", 1.0)),
            dz=float(getattr(grid, "dz", 1.0)),
            requested_nx=int(requested_nx if requested_nx is not None else nx),
            requested_ny=int(requested_ny if requested_ny is not None else ny),
            requested_nz=int(requested_nz if requested_nz is not None else nz),
            nx=nx,
            ny=ny,
            nz=nz,
            is_complete=(
                (requested_nx is None or nx == requested_nx)
                and (requested_ny is None or ny == requested_ny)
                and (requested_nz is None or nz == requested_nz)
            ),
        )

    def get_grid(
        self,
        center_latitude: float,
        center_longitude: float,
        center_depth_km: float,
        max_depth_km: float,
        max_distance_km: float,
        grid_latitude_spacing_km: float,
        grid_longitude_spacing_km: float,
        grid_depth_spacing_km: float,
        phase: str = "P",
        **kwargs: Any,
    ) -> GridSliceData:
        """Return a local volumetric grid slice for the requested phase."""
        phase_code = self._normalize_phase(phase)

        # For TVEL we can generate a local NLL grid with the requested geometry.
        if self.is_tvel:
            if self._tvel_model is None:
                self._tvel_model = TvelModel(self.model_path)

            requested_nx = (
                int(round(2.0 * max_distance_km / grid_longitude_spacing_km)) + 1
            )
            requested_ny = (
                int(round(2.0 * max_distance_km / grid_latitude_spacing_km)) + 1
            )
            requested_nz = int(round(max_depth_km / grid_depth_spacing_km)) + 1
            cache_key = (
                phase_code,
                f"{center_latitude:.5f}_{center_longitude:.5f}_{center_depth_km:.3f}_"
                f"{max_depth_km:.3f}_{max_distance_km:.3f}_{grid_latitude_spacing_km:.3f}_"
                f"{grid_longitude_spacing_km:.3f}_{grid_depth_spacing_km:.3f}",
            )
            cached = self._wrapped_phase_grids.get(cache_key)
            if cached is not None:
                return cached

            nll_grid = create_nll_grid_from_tvel(
                self.model_path,
                phase=phase_code,
                nx=requested_nx,
                ny=requested_ny,
                nz=requested_nz,
                dx=grid_longitude_spacing_km,
                dy=grid_latitude_spacing_km,
                dz=grid_depth_spacing_km,
                x_orig=-max_distance_km,
                y_orig=-max_distance_km,
                z_orig=0.0,
                orig_lat=center_latitude,
                orig_lon=center_longitude,
                proj_name="SIMPLE",
                **self._normalize_grid_kwargs(kwargs),
            )
            wrapped = self._wrap_nll_grid(
                nll_grid,
                requested_nx=requested_nx,
                requested_ny=requested_ny,
                requested_nz=requested_nz,
            )
            self._wrapped_phase_grids[cache_key] = wrapped
            return wrapped

        if self._nll_grid is None:
            raise RuntimeError("No NLLGrid loaded")

        # For existing NLL grids, return the full grid wrapped in the common interface.
        cache_key = (phase_code, "full")
        cached = self._wrapped_phase_grids.get(cache_key)
        if cached is not None:
            return cached

        wrapped = self._wrap_nll_grid(self._nll_grid)
        self._wrapped_phase_grids[cache_key] = wrapped
        return wrapped
