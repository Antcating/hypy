"""NetCDF-backed 3D velocity grid loader.

The loader normalizes coordinate orientation and transposes source arrays into
the local (x, y, depth) layout expected by the volumetric propagation code.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

from hypy.model import VelocityModel
from hypy.utils.geo import gps2dist_azimuth_nb
from hypy.utils.seismic import normalize_phase

from .. import GridSliceData


class NCGrid(VelocityModel):
    """3D velocity grid support using NetCDF4 grids.

    Parameters
    ----------
    model_path : Path | str
        Path to the NetCDF file containing the velocity grid data. The file must
        include the required coordinate variables and phase grids as described above.
    Note
    ---------------
    Expected coordinate variables:
      - latitude
      - longitude
      - depth

    Expected phase variable names:
      - pwave_model
      - swave_model

    The source NetCDF files do not need to share a fixed axis order. The loader
    inspects the variable dimensions, normalizes any reversed coordinates, and
    then returns a local cube in the common (x, y, depth) order expected by the
    volumetric propagation code.
    """

    feature_flags = frozenset(
        {
            "model.volumetric",
            "model.grid.3d",
            "format.netcdf",
            "phase.p",
            "phase.s",
        }
    )

    def __init__(self, model_path: Path | str):
        model_path = Path(model_path).expanduser().resolve()
        super().__init__(model_path=model_path, model_name=model_path.stem, dim=3)
        self._calculated_grids: dict[str, GridSliceData] = {}

    @staticmethod
    def _nearest_index(values: np.ndarray, value: float) -> int:
        return int(np.argmin(np.abs(values - value)))

    @staticmethod
    def _axis_index(dims: tuple[str, ...], needle: str) -> int:
        for index, dim_name in enumerate(dims):
            if needle in dim_name:
                return index
        raise ValueError(f"Could not find axis matching {needle!r} in {dims!r}")

    def _generate_grid_key(
        self,
        phase: str,
        center_latitude: float,
        center_longitude: float,
        center_depth_km: float,
        max_depth_km: float,
        max_distance_km: float,
        grid_latitude_spacing_km: float,
        grid_longitude_spacing_km: float,
        grid_depth_spacing_km: float,
    ) -> str:
        return (
            f"{phase}_"
            f"{center_latitude:.5f}_{center_longitude:.5f}_{center_depth_km:.3f}_"
            f"{max_depth_km:.3f}_{max_distance_km:.3f}_"
            f"{grid_latitude_spacing_km:.3f}_{grid_longitude_spacing_km:.3f}_{grid_depth_spacing_km:.3f}"
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
        **kwargs,
    ) -> GridSliceData:
        """Return a receiver-centered local cube from the NetCDF velocity model."""
        del kwargs  # Included for compatibility with the grid provider interface.
        phase_code = normalize_phase(phase)
        key = self._generate_grid_key(
            phase=phase_code,
            center_latitude=center_latitude,
            center_longitude=center_longitude,
            center_depth_km=center_depth_km,
            max_depth_km=max_depth_km,
            max_distance_km=max_distance_km,
            grid_latitude_spacing_km=grid_latitude_spacing_km,
            grid_longitude_spacing_km=grid_longitude_spacing_km,
            grid_depth_spacing_km=grid_depth_spacing_km,
        )
        cached = self._calculated_grids.get(key)
        if cached is not None:
            return cached

        requested_nx = int(round(2.0 * max_distance_km / grid_longitude_spacing_km)) + 1
        requested_ny = int(round(2.0 * max_distance_km / grid_latitude_spacing_km)) + 1
        requested_nz = int(round(max_depth_km / grid_depth_spacing_km)) + 1

        x_orig_req = -float(max_distance_km)
        y_orig_req = -float(max_distance_km)
        z_orig_req = 0.0

        with netCDF4.Dataset(self.model_path.as_posix(), "r") as nc:
            try:
                # Keep the raw coordinate arrays; we normalize orientation below
                # so files with descending axes remain usable.
                latitudes = nc.variables["latitude"][:]
                longitudes = nc.variables["longitude"][:]
                depths = nc.variables["depth"][:]
                phase_grid_name = f"{phase_code.lower()}wave_model"
                phase_var = nc.variables[phase_grid_name]
                raw_grid = np.asarray(phase_var[:], dtype=np.float32)
            except KeyError as exc:
                raise ValueError(
                    f"Grid not found in NetCDF file for phase: {phase_code}"
                ) from exc

            if not (latitudes.min() <= center_latitude <= latitudes.max()):
                raise ValueError(
                    f"Center latitude {center_latitude} out of bounds "
                    f"({latitudes.min()} to {latitudes.max()})"
                )
            if not (longitudes.min() <= center_longitude <= longitudes.max()):
                raise ValueError(
                    f"Center longitude {center_longitude} out of bounds "
                    f"({longitudes.min()} to {longitudes.max()})"
                )
            if not (depths.min() <= center_depth_km <= depths.max()):
                raise ValueError(
                    f"Center depth {center_depth_km} out of bounds "
                    f"({depths.min()} to {depths.max()})"
                )

            latitude_resolution_km = (
                gps2dist_azimuth_nb(
                    latitudes[0], center_longitude, latitudes[1], center_longitude
                )[0]
                / 1000.0
            )
            longitude_resolution_km = (
                gps2dist_azimuth_nb(
                    center_latitude, longitudes[0], center_latitude, longitudes[1]
                )[0]
                / 1000.0
            )
            depth_resolution_km = float(np.abs(depths[1] - depths[0]))

            dims = tuple(str(d).lower() for d in getattr(phase_var, "dimensions", ()))
            # The NetCDF variable can declare axes in any order; map them by name
            # so the output is always transposed into lon/lat/depth order.
            if raw_grid.ndim != 3:
                raise ValueError(
                    f"Unsupported NetCDF grid layout for {phase_grid_name!r}; "
                    f"dimensions={dims!r}"
                )

            lon_axis = self._axis_index(dims, "lon")
            lat_axis = self._axis_index(dims, "lat")
            depth_axis = self._axis_index(dims, "dep")
            if "z" in dims and "dep" not in dims:
                depth_axis = self._axis_index(dims, "z")

            if longitudes[0] > longitudes[-1]:
                # Downstream slicing assumes ascending coordinates.
                longitudes = longitudes[::-1]
                raw_grid = np.flip(raw_grid, axis=lon_axis)

            if latitudes[0] > latitudes[-1]:
                # Same normalization for latitude.
                latitudes = latitudes[::-1]
                raw_grid = np.flip(raw_grid, axis=lat_axis)

            if depths[0] > depths[-1]:
                # Depth is usually increasing downward, but we accept either form.
                depths = depths[::-1]
                raw_grid = np.flip(raw_grid, axis=depth_axis)

            # Reorder the data cube to the canonical (lon, lat, depth) layout used
            # by the local-grid projection and clipping logic below.
            src = np.transpose(
                raw_grid,
                axes=(lon_axis, lat_axis, depth_axis),
            )

        center_lon_idx = self._nearest_index(longitudes, center_longitude)
        center_lat_idx = self._nearest_index(latitudes, center_latitude)
        surface_depth_idx = self._nearest_index(depths, 0.0)

        half_nx = requested_nx // 2
        half_ny = requested_ny // 2

        src_x0 = max(0, center_lon_idx - half_nx)
        src_x1 = min(src.shape[0], center_lon_idx + half_nx + 1)
        src_y0 = max(0, center_lat_idx - half_ny)
        src_y1 = min(src.shape[1], center_lat_idx + half_ny + 1)
        src_z0 = max(0, surface_depth_idx)
        src_z1 = min(src.shape[2], surface_depth_idx + requested_nz)

        # Slice the centered local cube and record whether any side was clipped.
        array = np.asarray(
            src[src_x0:src_x1, src_y0:src_y1, src_z0:src_z1], dtype=np.float32
        )
        actual_nx, actual_ny, actual_nz = array.shape

        clipped_x_min = src_x0 > center_lon_idx - half_nx
        clipped_x_max = src_x1 < center_lon_idx + half_nx + 1
        clipped_y_min = src_y0 > center_lat_idx - half_ny
        clipped_y_max = src_y1 < center_lat_idx + half_ny + 1
        clipped_z_min = src_z0 > surface_depth_idx
        clipped_z_max = src_z1 < surface_depth_idx + requested_nz

        # The origins are expressed in the local Cartesian frame used by the
        # propagation code, not in geographic coordinates.
        x_orig = (
            x_orig_req
            + max(0, (src_x0 - (center_lon_idx - half_nx))) * grid_longitude_spacing_km
        )
        y_orig = (
            y_orig_req
            + max(0, (src_y0 - (center_lat_idx - half_ny))) * grid_latitude_spacing_km
        )
        z_orig = (
            z_orig_req + max(0, (src_z0 - surface_depth_idx)) * grid_depth_spacing_km
        )

        result = GridSliceData(
            array=array,
            orig_lat=float(center_latitude),
            orig_lon=float(center_longitude),
            x_orig=float(x_orig),
            y_orig=float(y_orig),
            z_orig=float(z_orig),
            dx=float(
                longitude_resolution_km if actual_nx > 1 else grid_longitude_spacing_km
            ),
            dy=float(
                latitude_resolution_km if actual_ny > 1 else grid_latitude_spacing_km
            ),
            dz=float(depth_resolution_km if actual_nz > 1 else grid_depth_spacing_km),
            requested_nx=requested_nx,
            requested_ny=requested_ny,
            requested_nz=requested_nz,
            nx=actual_nx,
            ny=actual_ny,
            nz=actual_nz,
            is_complete=(
                actual_nx == requested_nx
                and actual_ny == requested_ny
                and actual_nz == requested_nz
            ),
            clipped_x_min=clipped_x_min,
            clipped_x_max=clipped_x_max,
            clipped_y_min=clipped_y_min,
            clipped_y_max=clipped_y_max,
            clipped_z_min=clipped_z_min,
            clipped_z_max=clipped_z_max,
        )
        self._calculated_grids[key] = result
        return result
