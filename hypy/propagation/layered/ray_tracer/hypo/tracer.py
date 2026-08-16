"""Fast local 1-D ray tracer backed by :class:`hypy.model.layered.tvel.TvelModel`.

This module implements a simple local-earth, horizontally layered first-arrival
tracer.

Algorithm overview
------------------
For one source depth, epicentral distance, and phase, the tracer computes the
fastest of two families of arrivals:

* a direct/upgoing ray that leaves the source, bends through the horizontal
  layers according to Snell's law, and reaches the surface receiver;
* critically refracted head-wave branches along deeper, faster layers.

This is the same broad class of local 1-D travel-time logic used by classic
local-earthquake locators.  The per-point solver and the batch grid builder are
compiled with numba; see :mod:`hypy.propagation.layered.ray_tracer.hypo._kernels`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from hypy.logger import logger
from hypy.model.layered.tvel import TvelModel
from hypy.propagation.layered.ray_tracer import LayeredRayTracer
from hypy.propagation.layered.ray_tracer.hypo._kernels import (
    build_table,
    solve_one_point,
)
from hypy.utils.seismic import normalize_phase


class HypoRayTracer(LayeredRayTracer):
    """Fast first-arrival ray tracer for TVEL 1-D local P/S models.

    The spherical model is mapped onto a flat-layered geometry with the standard
    earth-flattening transform (``z_flat = R * ln(R / (R - z))`` and
    ``v_flat = v * R / (R - z)``) and resampled at ``layer_step_km`` intervals
    with native discontinuity depths forced into the edge list so interfaces are
    not snapped onto the step grid.  The stack of thin homogeneous flat layers
    reproduces the spherical diving-ray and head-wave travel times of the
    full-physics backends (e.g. TauP/pyrocko) to well within a millisecond-level
    relative tolerance, while remaining fast enough for location iterations.
    Epicentral distance is a surface arc length and is unchanged by the
    transform, and takeoff angles are preserved by earth flattening.

    Public methods return the same hypy-format arrivals as the other layered
    ray tracers: travel time in seconds and takeoff angle in degrees.  Internally
    the tracer uses the horizontal slowness/ray parameter ``p``:

        p = sin(theta) / v

    where ``theta`` is measured from vertical in a layer with velocity ``v``.
    A single ``p`` is shared by all segments of a ray in a 1-D layered medium.

    The per-point solve and the :meth:`build_travel_time_table` grid builder run
    as compiled numba kernels.

    Additional keyword arguments are forwarded to ``LayeredRayTracer``.  Use
    ``arrival_cache_size`` and ``arrival_cache_precision_km`` there to configure
    the shared layered-arrival cache
    """

    feature_flags = frozenset(
        {
            "propagation.layered",
            "propagation.table",
            "derivative.analytical",
            "derivative.numerical",
        }
    )
    required_model_features = frozenset({"model.profile.1d", "model.profile.1d"})
    name = "hypo"

    def __init__(
        self,
        model: TvelModel,
        *,
        max_distance_km: float = 300.0,
        max_depth_km: float = 50.0,
        layer_step_km: float = 1.0,
        eps: float = 1.0e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)

        if max_distance_km <= 0.0 or max_depth_km <= 0.0:
            raise ValueError("max_distance_km and max_depth_km must be positive.")
        if layer_step_km <= 0.0:
            raise ValueError("layer_step_km must be positive.")

        if type(model) is not TvelModel:
            raise ValueError(
                f"HypoRayTracer requires a TvelModel instance, but got {type(model)}."
            )
        self.velocity_model = model

        self.max_distance_km = max_distance_km
        self.max_depth_km = max_depth_km
        self.layer_step_km = float(layer_step_km)
        # Radius of the (spherical) planet used for the earth-flattening
        # transform that maps the spherical model onto the flat-layered geometry
        # this tracer solves in.
        self._radius_km = model.radius_of_planet_km
        self._phase_profile_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._eps = float(eps)

    def get_arrival_1d(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ) -> tuple[float, float]:
        """Return first-arrival time and takeoff angle for one 1-D geometry.

        Negative distance is mirrored to positive distance because this 1-D
        calculation depends only on offset magnitude, not azimuth.
        """
        phase_key = normalize_phase(phase)
        if not np.isfinite(arc_distance_km) or not np.isfinite(source_depth_km):
            raise ValueError("Distance and depth must be finite.")
        arc_distance_km = float(abs(arc_distance_km))
        if source_depth_km < 0.0:
            raise ValueError("source_depth_km must be non-negative.")
        if arc_distance_km > self.max_distance_km:
            raise ValueError(
                f"Distance {arc_distance_km:.3f} km exceeds configured local "
                f"coverage {self.max_distance_km:.3f} km. Increase max_distance_km."
            )
        if source_depth_km > self.max_depth_km:
            raise ValueError(
                f"Depth {source_depth_km:.3f} km exceeds configured local "
                f"coverage {self.max_depth_km:.3f} km. Increase max_depth_km."
            )

        distance_km = float(arc_distance_km)
        depth_km = float(source_depth_km)

        # Arrivals are often requested repeatedly during location iterations.
        cache_key = self._arrival_cache_key(phase_key, distance_km, depth_km)
        cached = self._arrival_cache_get(cache_key)
        if cached is not None:
            return cached

        tops, velocities = self._phase_profile(phase_key)
        # ``v_source`` is the gradient-interpolated velocity at the source depth;
        # the kernel uses it only for the takeoff angle and the head-wave check.
        # Both the depth and the velocity are earth-flattened so they are
        # consistent with the flattened layer profile. The epicentral distance
        # is a surface arc length and is left unchanged by the transform, and
        # takeoff angles are preserved by earth flattening.
        v_source = self._flatten_velocity(
            float(self.velocity_model.get_velocity_1d(depth_km, phase_key)),
            depth_km,
        )
        depth_flat = float(self._flatten_depth(depth_km))
        time_s, angle_deg = solve_one_point(
            tops,
            velocities,
            float(v_source),
            depth_flat,
            distance_km,
            float(tops[-1]),
            self._eps,
        )
        result = (float(time_s), float(angle_deg))
        self._arrival_cache_put(cache_key, result)
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
    ) -> tuple[float, Optional[float], Optional[float], float]:
        """Return hypy-format arrival parameters for a source-receiver pair.

        This method wraps the local 1-D calculation with the geographic work
        needed by the public propagation interface.  Latitude/longitude are used
        only to compute epicentral distance and azimuth; the travel-time physics
        is purely 1-D without any geographic corrections.
        """
        arc_distance_km, azimuth_deg, _ = self._compute_arc_params(
            source_latitude,
            source_longitude,
            receiver_latitude,
            receiver_longitude,
        )

        # Elevation is folded into an effective source depth relative to the
        # receiver.  A positive receiver elevation increases the source-receiver
        # vertical separation by ``elevation / 1000`` km.
        eff_source_depth_km = source_depth_km + receiver_elevation_m / 1000.0

        if max_distance_km is not None and arc_distance_km > max_distance_km:
            logger.info(
                "Station at distance %.2f km exceeds max_distance %.2f km. Skipping.",
                arc_distance_km,
                max_distance_km,
            )
            return arc_distance_km, None, None, azimuth_deg

        time_s, takeoff_angle_deg = self.get_arrival_1d(
            arc_distance_km=arc_distance_km,
            source_depth_km=eff_source_depth_km,
            phase=phase,
        )
        return arc_distance_km, time_s, takeoff_angle_deg, azimuth_deg

    def build_travel_time_table(
        self,
        phase: str,
        depth_grid: np.ndarray,
        distance_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fill a full ``(depth, distance)`` grid in one parallel compiled call.

        This is the fast path for travel-time table generation: instead of
        calling :meth:`get_arrival_1d` per grid node, the whole grid is solved in
        a single numba call parallelized over the depth axis.

        Parameters
        ----------
        phase : str
            Seismic phase (``"P"`` or ``"S"``).
        depth_grid : np.ndarray
            1-D array of source depths (km).
        distance_grid : np.ndarray
            1-D array of epicentral distances (km).

        Returns
        -------
        tuple of np.ndarray
            ``(travel_time_s, takeoff_angle_deg)`` arrays of shape
            ``(len(depth_grid), len(distance_grid))``.
        """
        phase_key = normalize_phase(phase)
        tops, velocities = self._phase_profile(phase_key)
        depth_grid = np.ascontiguousarray(depth_grid, dtype=np.float64)
        distance_grid = np.ascontiguousarray(distance_grid, dtype=np.float64)
        # Earth-flatten the source depths and velocities so they match the
        # flattened layer profile; distances are surface arc lengths and are
        # left unchanged.
        v_source_grid = np.ascontiguousarray(
            [
                self._flatten_velocity(
                    self.velocity_model.get_velocity_1d(float(z), phase_key),
                    float(z),
                )
                for z in depth_grid
            ],
            dtype=np.float64,
        )
        depth_grid_flat = np.ascontiguousarray(
            self._flatten_depth(depth_grid), dtype=np.float64
        )
        return build_table(
            tops,
            velocities,
            v_source_grid,
            depth_grid_flat,
            distance_grid,
            float(tops[-1]),
            self._eps,
        )

    def _flatten_depth(self, depth_km: float | np.ndarray):
        """Earth-flattening transform for depth: spherical -> flat.

        ``z_flat = R * ln(R / (R - z))`` maps a spherical-earth depth onto the
        equivalent flat-earth depth so the flat-layer kernel reproduces the
        spherical travel times used by the other layered backends.
        """
        r = self._radius_km
        return r * np.log(r / (r - depth_km))

    def _flatten_velocity(
        self, velocity: float | np.ndarray, depth_km: float | np.ndarray
    ):
        """Earth-flattening transform for velocity: ``v_flat = v * R / (R - z)``."""
        r = self._radius_km
        return velocity * r / (r - depth_km)

    def _profile_max_depth_km(self) -> float:
        """Real depth to which the flattened profile is resampled.

        The profile is resampled down to ``max_depth_km`` (the deepest source
        depth handled by the tracer).  For the local/regional distances this
        tracer targets, the first-arrival refracted rays turn in the uppermost
        mantle at or above this depth, so extending the profile deeper is
        unnecessary and only slows the O(n^2) head-wave search.

        Note: ``max_depth_km`` must be set below the dominant crustal refractor
        (e.g. the Moho); a value above it would drop the refracted (Pn/Sn)
        first arrivals at regional distances.  The bound is also clamped well
        inside the planet radius to keep the earth-flattening transform finite.
        """
        return float(min(self.max_depth_km, 0.9 * self._radius_km))

    def _profile_edges(self, phase: str) -> np.ndarray:
        """Build real-depth layer edges that preserve model interfaces.

        A uniform ``layer_step_km`` grid alone snaps discontinuities onto the
        step (e.g. 2.6 km → 3.0 km), inventing extra slow rock and biasing
        travel times.  Merge the model's native layer-top depths into the grid
        so interfaces stay exact while the average resolution stays ~``step``.
        """
        step = self.layer_step_km
        zmax = self._profile_max_depth_km()
        grid = np.arange(0.0, zmax + step, step, dtype=np.float64)

        phase_key = normalize_phase(phase)
        try:
            iface_tops, _ = self.velocity_model.get_layer_top_profile_1d(phase_key)
        except (ValueError, AttributeError):
            iface_tops = np.array([], dtype=np.float64)

        interfaces = np.asarray(iface_tops, dtype=np.float64)
        interfaces = interfaces[(interfaces > 0.0) & (interfaces < zmax)]
        edges = np.unique(np.concatenate([grid, interfaces, [zmax]]))
        edges.sort()
        if edges.size < 2 or edges[0] != 0.0:
            raise ValueError(
                "Failed to build a valid layer-edge profile starting at 0 km."
            )
        return edges

    def _phase_profile(self, phase: str) -> tuple[np.ndarray, np.ndarray]:
        """Return cached, C-contiguous earth-flattened layer profile.

        The model is resampled at ``layer_step_km`` intervals with native
        discontinuity depths forced into the edge list, then mapped through the
        earth-flattening transform so the homogeneous flat-layer kernel
        approximates the spherical diving rays (and head waves) computed by the
        full-physics backends.  The arrays are ``float64`` and contiguous so
        they can be passed directly into the numba kernels without per-call
        copies.
        """
        phase_key = normalize_phase(phase)
        cached = self._phase_profile_cache.get(phase_key)
        if cached is not None:
            return cached

        edges = self._profile_edges(phase_key)
        tops_real = edges[:-1]
        bots_real = edges[1:]
        # Sample just below each layer top so a preserved interface keeps the
        # velocity of the layer beneath it (TauP/evaluate_below convention).
        sample_eps = min(1.0e-6, 0.01 * self.layer_step_km)
        sample_depths = np.minimum(tops_real + sample_eps, bots_real - sample_eps)
        # Degenerate (near-zero thickness) slabs: fall back to the top depth.
        too_thin = bots_real - tops_real <= 2.0 * sample_eps
        sample_depths = np.where(too_thin, tops_real, sample_depths)
        vels_real = np.array(
            [
                self.velocity_model.get_velocity_1d(float(z), phase_key)
                for z in sample_depths
            ],
            dtype=np.float64,
        )
        if np.any(vels_real <= 0.0):
            raise ValueError(
                f"All {phase.upper()} velocities must be positive for local tracing."
            )

        mids_real = 0.5 * (tops_real + bots_real)
        tops = np.ascontiguousarray(self._flatten_depth(tops_real), dtype=np.float64)
        velocities = np.ascontiguousarray(
            self._flatten_velocity(vels_real, mids_real), dtype=np.float64
        )
        self._phase_profile_cache[phase_key] = (tops, velocities)
        return tops, velocities
