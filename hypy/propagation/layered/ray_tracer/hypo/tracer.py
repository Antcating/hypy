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
local-earthquake locators.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from hypy.logger import logger
from hypy.model.layered.tvel import TvelModel
from hypy.propagation.layered.ray_tracer import LayeredRayTracer


class HypoRayTracer(LayeredRayTracer):
    """Fast first-arrival ray tracer for TVEL 1-D local P/S models.

    The spherical model is mapped onto a flat-layered geometry with the standard
    earth-flattening transform (``z_flat = R * ln(R / (R - z))`` and
    ``v_flat = v * R / (R - z)``) and finely resampled at ``layer_step_km``
    intervals.  The stack of thin homogeneous flat layers reproduces the
    spherical diving-ray and head-wave travel times of the full-physics backends
    (e.g. TauP/pyrocko) to well within a millisecond-level relative tolerance,
    while remaining fast enough for location iterations.  Epicentral distance is
    a surface arc length and is unchanged by the transform, and takeoff angles
    are preserved by earth flattening.

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

        phase_key = self._phase_key(phase)
        distance_km = float(arc_distance_km)
        depth_km = float(source_depth_km)

        # Arrivals are often requested repeatedly during location iterations.
        cache_key = self._arrival_cache_key(phase_key, distance_km, depth_km)
        cached = self._arrival_cache_get(cache_key)
        if cached is not None:
            return cached

        tops, velocities = self._phase_profile(phase_key)
        result = self._solve_one_point(
            tops=tops,
            velocities_km_s=velocities,
            phase=phase_key,
            source_depth_km=depth_km,
            distance_km=distance_km,
        )
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

    @staticmethod
    def _phase_key(phase: str) -> str:
        """Normalize phase names to the two velocity profiles supported here."""
        phase_lower = phase.strip().lower()
        if phase_lower.startswith("p"):
            return "p"
        if phase_lower.startswith("s"):
            return "s"
        raise ValueError(
            f"Unsupported phase: {phase}. Supported phases are 'P' and 'S'."
        )

    def _phase_profile(self, phase: str) -> tuple[np.ndarray, np.ndarray]:
        """Return cached layer-top depths and velocities for one phase."""
        phase_key = self._phase_key(phase)
        cached = self._phase_profile_cache.get(phase_key)
        if cached is not None:
            return cached

        tops, velocities = self.velocity_model.get_layer_top_profile_1d(
            phase_key,
            drop_non_positive_velocity=True,
        )
        self._phase_profile_cache[phase_key] = (tops, velocities)
        return tops, velocities

    def _solve_one_point(
        self,
        *,
        tops: np.ndarray,
        velocities_km_s: np.ndarray,
        phase: str,
        source_depth_km: float,
        distance_km: float,
    ) -> tuple[float, float]:
        """Compute the first P/S arrival for one depth-distance pair.

        This is the central branch-selection routine.  It first computes the
        direct/upgoing arrival, then tests each physically possible critically
        refracted branch and keeps whichever arrival is fastest.

        The returned takeoff angle follows the convention:
        degrees from the downward vertical at the source. A vertically
        upgoing ray is therefore 180 degrees, a horizontal ray is
        near 90 degrees, and downgoing refracted branches have smaller angles.
        """
        source_layer = self._layer_index(tops, source_depth_km)
        source_velocity = float(
            self.velocity_model.get_velocity_1d(source_depth_km, phase)
        )

        # Start with the direct branch as the default winner.  In a slow-over-fast
        # layered model this branch dominates at short offsets; deeper head waves
        # may overtake it at larger offsets.
        best_time, best_angle = self._direct_point(
            tops=tops,
            velocities=velocities_km_s,
            source_depth_km=source_depth_km,
            source_velocity=source_velocity,
            distance_km=distance_km,
        )

        # Test homogeneous-layer critically refracted first-arrival branches.
        #
        # A head wave can only run along a layer whose velocity is faster than
        # all layers it must pass through.  For each candidate refractor, the
        # ray parameter is fixed at the critical value p = 1 / v_refractor.
        # The ray travels from source to the interface, propagates horizontally
        # along the top of the refractor, and returns to the surface receiver.
        start_refractor = source_layer + 1
        source_on_layer_top = (
            abs(source_depth_km - float(tops[source_layer])) < 1.0e-9
            and source_layer > 0
        )
        if source_on_layer_top:
            # If the source lies exactly on an interface, allow the layer below
            # that interface to be considered as the refractor.  Without this
            # special case we would skip a valid downgoing critically refracted
            # branch that starts directly at the discontinuity.
            start_refractor = source_layer

        for refractor_layer in range(start_refractor, len(velocities_km_s)):
            interface_depth = float(tops[refractor_layer])

            # Local guard: do not scan the deep mantle/core for local offsets.
            if (
                interface_depth
                > source_depth_km + max(2.0 * distance_km, self.max_depth_km) + 50.0
            ):
                break

            refractor_velocity = float(velocities_km_s[refractor_layer])
            fastest_above = float(np.max(velocities_km_s[:refractor_layer]))

            # Critical refraction into this layer is possible only when the
            # candidate refractor is strictly faster than the layers above it.
            # Otherwise the critical angle is not real, or an earlier faster
            # layer would already provide the controlling head-wave branch.
            if source_on_layer_top and refractor_layer == source_layer:
                if refractor_velocity <= fastest_above + 1.0e-10:
                    continue
            elif refractor_velocity <= max(fastest_above, source_velocity) + 1.0e-10:
                continue

            # Critical ray parameter.  Because p is horizontal slowness, the
            # horizontal part of the head wave contributes distance / velocity,
            # written here as ``distance * p``.
            p = 1.0 / refractor_velocity
            x_legs, t_legs = self._head_wave_legs(
                tops=tops,
                velocities=velocities_km_s,
                source_depth_km=source_depth_km,
                interface_depth_km=interface_depth,
                refractor_layer=refractor_layer,
                p=p,
            )
            if not np.isfinite(x_legs) or distance_km < x_legs:
                # ``x_legs`` is the minimum surface offset consumed by the two
                # dipping legs.  If the station is closer than that, there is no
                # room for a physical horizontal head-wave segment.
                continue

            head_time = t_legs + (distance_km - x_legs) * p
            if head_time < best_time:
                best_time = float(head_time)
                # For a critically refracted branch the takeoff angle at the
                # source is determined by Snell's law in the source layer:
                # sin(theta) = p * v_source.  This angle is measured from the
                # downward vertical, matching the downgoing branch convention.
                best_angle = float(
                    np.degrees(np.arcsin(np.clip(p * source_velocity, 0.0, 1.0)))
                )

        return float(best_time), float(best_angle)

    def _direct_point(
        self,
        *,
        tops: np.ndarray,
        velocities: np.ndarray,
        source_depth_km: float,
        source_velocity: float,
        distance_km: float,
    ) -> tuple[float, float]:
        """Direct upgoing branch from source depth to the surface.

        The direct branch is solved in ray-parameter space.  For a fixed ``p``,
        each crossed layer contributes a horizontal distance and travel time.
        The correct direct ray is the ``p`` whose summed horizontal distance
        equals the requested source-receiver offset.

        In a homogeneous half-space this reduces to the familiar straight-line
        solution.  In multiple horizontal layers it bends at each interface while
        preserving the same horizontal slowness ``p``.
        """
        if source_depth_km <= self._eps:
            # Surface source: the direct ray has no vertical layer crossings.
            time_s = distance_km / float(velocities[0])
            angle_deg = 180.0 if distance_km <= self._eps else 90.0
            return float(time_s), angle_deg

        # Only layers between the free surface and source depth participate in
        # the direct upgoing path.
        seg_layers, seg_thicknesses = self._segments_between(
            tops=tops,
            z0=0.0,
            z1=source_depth_km,
        )
        crossed_velocities = velocities[seg_layers]
        p_crit = 1.0 / float(np.max(crossed_velocities))

        if distance_km <= self._eps:
            # Zero-offset receiver: the direct ray is vertical, so time is just
            # the sum of vertical thickness / velocity through crossed layers.
            inv_v = 1.0 / crossed_velocities
            return float(np.sum(seg_thicknesses * inv_v)), 180.0

        # Binary-search p between vertical (p = 0) and the smallest critical
        # slowness among crossed layers.  Horizontal distance increases
        # monotonically with p, so this is stable and cheap.
        lo = 0.0
        hi = p_crit * (1.0 - self._eps)
        p = 0.0

        for _ in range(48):
            p = 0.5 * (lo + hi)
            x_mid, _t_mid = self._x_t_for_p(
                velocities=crossed_velocities,
                thicknesses=seg_thicknesses,
                p=p,
            )
            if x_mid < distance_km:
                lo = p
            else:
                hi = p

        p = 0.5 * (lo + hi)
        _x_val, time_s = self._x_t_for_p(
            velocities=crossed_velocities,
            thicknesses=seg_thicknesses,
            p=p,
        )
        emergence_from_upward_vertical = float(
            np.degrees(np.arcsin(np.clip(p * source_velocity, 0.0, 1.0)))
        )
        # ``emergence_from_upward_vertical`` is the geometric angle of the
        # upgoing ray.  The public takeoff convention is measured from the
        # downward vertical, so an upgoing ray is reported as 180 - emergence.
        takeoff_from_downward_vertical = 180.0 - emergence_from_upward_vertical
        return float(time_s), takeoff_from_downward_vertical

    @staticmethod
    def _x_t_for_p(
        *,
        velocities: np.ndarray,
        thicknesses: np.ndarray,
        p: float,
    ) -> tuple[float, float]:
        """Return horizontal distance and time for layer segments at fixed ``p``.

        For one layer of thickness ``h`` and velocity ``v``:

        * vertical slowness is ``eta = sqrt(1 / v**2 - p**2)``;
        * horizontal distance is ``h * p / eta``;
        * travel time is ``h / (v**2 * eta)``.

        These are standard flat-layer ray-parameter formulas.  Summing them over
        all crossed layers gives the total direct-leg distance and travel time.
        """
        inv_v = 1.0 / velocities
        eta = np.sqrt(np.maximum(inv_v * inv_v - p * p, 1.0e-300))
        x_km = np.sum(thicknesses * p / eta)
        t_s = np.sum(thicknesses * inv_v * inv_v / eta)
        return float(x_km), float(t_s)

    def _head_wave_legs(
        self,
        *,
        tops: np.ndarray,
        velocities: np.ndarray,
        source_depth_km: float,
        interface_depth_km: float,
        refractor_layer: int,
        p: float,
    ) -> tuple[float, float]:
        """Leg distance/time to and from a critically refracting interface.

        A head-wave arrival is split into three pieces:

        1. a slant leg from the source down/up to the refracting interface;
        2. a horizontal segment along the fast refractor;
        3. a slant leg from the interface back to the surface receiver.

        This helper computes only the two slant-leg totals.  The horizontal
        segment is added by ``_solve_one_point`` as
        ``(distance_km - x_legs) * p``.

        ``layers_up`` represents the receiver-side leg from the interface to the
        surface.  ``layers_down`` represents the source-side leg when the source
        is above the refractor.  If the source is at or below the candidate
        interface, there is no additional downgoing leg to add.
        """
        layers_up, thicknesses_up = self._segments_between(
            tops=tops,
            z0=0.0,
            z1=interface_depth_km,
        )
        if interface_depth_km > source_depth_km:
            layers_down, thicknesses_down = self._segments_between(
                tops=tops,
                z0=source_depth_km,
                z1=interface_depth_km,
            )
            layers = np.concatenate([layers_up, layers_down])
            thicknesses = np.concatenate([thicknesses_up, thicknesses_down])
        else:
            layers = layers_up
            thicknesses = thicknesses_up

        x_total = 0.0
        t_total = 0.0
        for layer, thickness in zip(layers, thicknesses):
            if int(layer) >= refractor_layer:
                # The critically refracted wave travels along the top of the
                # refractor, not through its interior as a finite-thickness leg.
                continue
            v = float(velocities[int(layer)])
            inv_v = 1.0 / v
            eta2 = inv_v * inv_v - p * p
            if eta2 <= 0.0:
                # Non-positive vertical slowness means this candidate p cannot
                # propagate through the overlying layer.  Treat the branch as
                # invalid so the caller can skip it.
                return np.inf, np.inf
            eta = float(np.sqrt(eta2))
            x_total += float(thickness) * p / eta
            t_total += float(thickness) * inv_v * inv_v / eta

        return x_total, t_total

    def _segments_between(
        self,
        *,
        tops: np.ndarray,
        z0: float,
        z1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Describe which layers are crossed between two depths.

        Returns parallel arrays:

        * layer indices into the ``tops``/velocity arrays;
        * the thickness crossed in each layer.

        The order is shallow to deep regardless of input order.  The ray formulas
        only need the set of crossed layer thicknesses, so upward and downward
        traversals can use the same representation.
        """
        if z1 < z0:
            z0, z1 = z1, z0
        if abs(z1 - z0) <= self._eps:
            return np.array([], dtype=int), np.array([], dtype=float)

        layers: list[int] = []
        thicknesses: list[float] = []
        start_layer = self._layer_index(tops, z0)
        end_layer = self._layer_index(tops, z1)
        n_layers = len(tops)

        for layer in range(start_layer, end_layer + 1):
            top = max(float(tops[layer]), z0)
            if layer + 1 < n_layers:
                bottom = min(float(tops[layer + 1]), z1)
            else:
                bottom = z1
            thickness = bottom - top
            if thickness > self._eps:
                layers.append(layer)
                thicknesses.append(thickness)

        return np.asarray(layers, dtype=int), np.asarray(thicknesses, dtype=float)

    @staticmethod
    def _layer_index(tops: np.ndarray, depth_km: float) -> int:
        """Return the layer containing ``depth_km``.

        ``tops`` contains monotonically increasing layer-top depths.  A depth
        exactly on an interface is assigned to the layer below that interface,
        which matches the velocity lookup convention used for layer-top samples.
        """
        idx = int(np.searchsorted(tops, depth_km, side="right") - 1)
        return max(0, min(idx, len(tops) - 1))
