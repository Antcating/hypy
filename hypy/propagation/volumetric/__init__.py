"""Volumetric propagation model base classes."""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from hypy.model import VelocityModel
from hypy.propagation.base import PropagationEngine
from hypy.utils.math import finite_difference_derivative

from ...logger import logger


class VolumetricPropagationEngine(PropagationEngine):
    """Base class for volumetric 3D propagation backends."""

    def __init__(self, model: Union[VelocityModel, str, Path], **kwargs: Any):
        # A path is loaded through the volumetric router so ``.tvel`` files
        # become laterally homogeneous 3D grids (NllGridModel) rather than the
        # 1D TvelModel that ``VelocityModel.create_model`` would return.
        if isinstance(model, (str, Path)):
            from hypy.model.volumetric import build_velocity_grid_provider

            model = build_velocity_grid_provider(model)
        super().__init__(model, **kwargs)

    @property
    def dim(self) -> int:
        """Return the dimensionality of the propagation model.

        For a volumetric 3D model, this is typically ``3``.
        """
        return 3

    @property
    def name(self) -> str:
        """Return a human-readable name for this propagation engine."""
        return f"VolumetricPropagationEngine({self.velocity_model.model_name})"

    def compute_travel_time_params(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_elevation_m: float,
        phase: str,
        delta: float | None = None,
        derivative_method: str = "numerical",
    ) -> tuple[
        float,
        float | None,
        tuple[float | None, float | None, float | None, float | None],
    ]:
        """Return distance, travel time, and derivatives for a source/receiver
        pair.

        When the path exceeds the engine's configured distance limit or
        falls outside the velocity grid, ``travel_time`` and the
        derivative tuple entries are ``None``.
        """
        arc_distance_km, travel_time_s, _, _ = self.get_arrival(
            source_latitude=source_latitude,
            source_longitude=source_longitude,
            source_depth_km=source_depth_km,
            receiver_latitude=receiver_latitude,
            receiver_longitude=receiver_longitude,
            receiver_elevation_m=receiver_elevation_m,
            phase=phase,
            max_distance_km=self.max_distance_km,
        )

        if travel_time_s is None:
            return arc_distance_km, None, (None, None, None, None)

        if derivative_method != "numerical":
            raise ValueError(
                f"Invalid mode: {derivative_method}. "
                "Volumetric engines only support numerical derivatives."
            )
        if delta is None:
            raise ValueError(
                "Delta must be provided for numerical derivative computation."
            )

        try:
            dt_dr, dt_dx, dt_dy, dt_dz = self.compute_derivatives(
                source_latitude=source_latitude,
                source_longitude=source_longitude,
                source_depth_km=source_depth_km,
                receiver_latitude=receiver_latitude,
                receiver_longitude=receiver_longitude,
                receiver_elevation_m=receiver_elevation_m,
                phase=phase,
                delta=delta,
                mode=derivative_method,
            )
        except ValueError as exc:
            logger.warning(
                "Unable to compute numerical derivatives for source=(%s, %s, %s) "
                "receiver=(%s, %s, %s) phase=%s: %s",
                source_latitude,
                source_longitude,
                source_depth_km,
                receiver_latitude,
                receiver_longitude,
                receiver_elevation_m,
                phase,
                exc,
            )
            return arc_distance_km, travel_time_s, (None, None, None, None)

        return arc_distance_km, travel_time_s, (dt_dr, dt_dx, dt_dy, dt_dz)

    def _compute_derivatives_numerical(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_elevation_m: float,
        phase: str,
        delta_km: float,
        method: str = "central",
    ) -> tuple[float, float, float, float]:
        """Compute spatial slowness derivatives with numerical finite
        differences.
        """

        def _lat_offset_deg(delta_km: float) -> float:
            return float(delta_km / 111.0)

        def _lon_offset_deg(lat_deg: float, delta_km: float) -> float:
            return float(delta_km / (111.320 * np.cos(np.deg2rad(lat_deg))))

        def _travel_time_north(d_km: float) -> float:
            return self.get_arrival(
                source_latitude=source_latitude + _lat_offset_deg(d_km),
                source_longitude=source_longitude,
                source_depth_km=source_depth_km,
                receiver_latitude=receiver_latitude,
                receiver_longitude=receiver_longitude,
                receiver_elevation_m=receiver_elevation_m,
                phase=phase,
            )[1]

        def _travel_time_east(d_km: float) -> float:
            return self.get_arrival(
                source_latitude=source_latitude,
                source_longitude=source_longitude
                + _lon_offset_deg(source_latitude, d_km),
                source_depth_km=source_depth_km,
                receiver_latitude=receiver_latitude,
                receiver_longitude=receiver_longitude,
                receiver_elevation_m=receiver_elevation_m,
                phase=phase,
            )[1]

        dt_dy = finite_difference_derivative(
            _travel_time_north,
            0.0,
            delta_km,
            method=method,
        )

        dt_dx = finite_difference_derivative(
            _travel_time_east,
            0.0,
            delta_km,
            method=method,
        )

        depth_method = method
        if method == "central" and source_depth_km - delta_km < 0.0:
            depth_method = "forward"

        dt_dz = finite_difference_derivative(
            lambda z: self.get_arrival(
                source_latitude=source_latitude,
                source_longitude=source_longitude,
                source_depth_km=z,
                receiver_latitude=receiver_latitude,
                receiver_longitude=receiver_longitude,
                receiver_elevation_m=receiver_elevation_m,
                phase=phase,
            )[1],
            source_depth_km,
            delta_km,
            method=depth_method,
        )

        dt_dr = 0.0

        return dt_dr, dt_dx, dt_dy, dt_dz

    def compute_derivatives(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_elevation_m: float,
        phase: str,
        delta: Optional[float] = 0.01,
        mode: str = "numerical",
    ) -> tuple[float, float, float, float]:
        """Compute spatial slowness derivatives using the configured mode.

        Volumetric backends only support numerical finite differences;
        there is no takeoff-angle analogue of the 1D analytical
        derivative.
        """
        if mode == "analytical":
            raise ValueError(
                "Analytical derivatives are not supported for volumetric engines."
            )
        if mode != "numerical":
            raise ValueError(f"Invalid mode: {mode}. Supported mode is numerical.")
        if delta is None:
            raise ValueError(
                "Delta must be provided for numerical derivative computation."
            )
        return self._compute_derivatives_numerical(
            source_latitude=source_latitude,
            source_longitude=source_longitude,
            source_depth_km=source_depth_km,
            receiver_latitude=receiver_latitude,
            receiver_longitude=receiver_longitude,
            receiver_elevation_m=receiver_elevation_m,
            phase=phase,
            delta_km=delta,
        )
