"""Shared volumetric velocity-model types and backend selection helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from hypy.utils.geo import gps2dist_azimuth_nb


@dataclass(frozen=True)
class GridSliceData:
    """Local Cartesian velocity cube and associated geometry metadata."""

    array: np.ndarray

    # Geographic origin of the local cartesian grid
    orig_lat: float
    orig_lon: float

    # Cartesian origin in km
    x_orig: float
    y_orig: float
    z_orig: float

    # Spacing in km
    dx: float
    dy: float
    dz: float

    # Requested extents
    requested_nx: int
    requested_ny: int
    requested_nz: int

    # Actual extents returned
    nx: int
    ny: int
    nz: int

    # Whether requested window was fully available in the source model
    is_complete: bool

    # Whether the source model had to be clipped at each side
    clipped_x_min: bool = False
    clipped_x_max: bool = False
    clipped_y_min: bool = False
    clipped_y_max: bool = False
    clipped_z_min: bool = False
    clipped_z_max: bool = False

    def project(self, lat: float, lon: float) -> tuple[float, float]:
        """Project latitude/longitude to local Cartesian x/y coordinates in
        km.
        """
        dist_m, az, _ = gps2dist_azimuth_nb(self.orig_lat, self.orig_lon, lat, lon)
        x = float(np.sin(np.deg2rad(az)) * dist_m) / 1000.0
        y = float(np.cos(np.deg2rad(az)) * dist_m) / 1000.0
        return x, y

    def contains_xyz(self, x: float, y: float, z: float) -> bool:
        """Return whether a Cartesian point is inside this cube bounds."""
        x_max = self.x_orig + (self.nx - 1) * self.dx
        y_max = self.y_orig + (self.ny - 1) * self.dy
        z_max = self.z_orig + (self.nz - 1) * self.dz
        return (
            self.x_orig <= x <= x_max
            and self.y_orig <= y <= y_max
            and self.z_orig <= z <= z_max
        )


class VelocityGridProvider(Protocol):
    """Protocol for backends that provide local volumetric velocity grids."""

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
        """Build or retrieve a local grid around a target geographic
        location.
        """
        ...


def build_velocity_grid_provider(model_path: str | Path) -> VelocityGridProvider:
    """Return a volumetric grid backend for the given model file."""
    model_path = Path(model_path).expanduser().resolve()
    suffix = model_path.suffix.lower()

    if suffix == ".nc":
        from .netcdf import NCGrid

        return NCGrid(model_path)

    # FIXME: It assumes that any non-netcdf model is a nllgrid.
    # We should ideally have a more robust way to determine the appropriate model type.
    from .nllgrid import NllGridModel

    return NllGridModel(model_path)


def __getattr__(name: str):
    """Lazily expose volumetric backend classes at package level."""
    if name == "NCGrid":
        from .netcdf import NCGrid

        return NCGrid
    if name == "NllGridModel":
        from .nllgrid import NllGridModel

        return NllGridModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GridSliceData",
    "VelocityGridProvider",
    "build_velocity_grid_provider",
    "NCGrid",
    "NllGridModel",
]
