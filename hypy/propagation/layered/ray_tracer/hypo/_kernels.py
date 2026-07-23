"""Numba-JIT kernels backing :class:`HypoRayTracer`.

These compiled functions implement the homogeneous-layer first-arrival logic
(direct/upgoing branch plus critically refracted head-wave branches) used by
:mod:`hypy.propagation.layered.ray_tracer.hypo.tracer`.  They are kept separate
from the public tracer class purely for readability; everything here stays in
numba ``nopython`` mode and operates on plain ``float64`` arrays and scalars.

Conventions (shared with the tracer)
-------------------------------------
* The model is a stack of horizontal homogeneous layers described by ``tops``
  (strictly increasing layer-top depths, ``tops[0] == 0``) and ``velocities``
  (km/s, one constant value per layer).
* A depth exactly on an interface belongs to the layer *below* it.
* ``v_source`` is the velocity at the source depth (gradient interpolated by the
  caller when the model has gradients); it controls only the takeoff-angle
  convention and the head-wave critical check.
* The takeoff angle is measured from the downward vertical at the source: a
  vertically upgoing direct ray is ``180``, a horizontal ray is near ``90``, and
  downgoing critically refracted branches have smaller angles.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _layer_index(tops: np.ndarray, depth: float) -> int:
    """Return the index of the layer containing ``depth``.

    Equivalent to ``searchsorted(tops, depth, side="right") - 1`` clamped to
    ``[0, n - 1]``.  A depth exactly on an interface is assigned to the layer
    below that interface.
    """
    n = tops.shape[0]
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if tops[mid] <= depth:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    if idx < 0:
        idx = 0
    if idx > n - 1:
        idx = n - 1
    return idx


@njit(cache=True)
def _direct_x_t(
    tops: np.ndarray,
    velocities: np.ndarray,
    source_depth: float,
    p: float,
    eps: float,
):
    """Horizontal distance and travel time of the direct leg at slowness ``p``.

    Sums the standard flat-layer ray-parameter contributions over every layer
    crossed between the free surface and ``source_depth``.
    """
    n = tops.shape[0]
    end = _layer_index(tops, source_depth)
    x = 0.0
    t = 0.0
    for layer in range(0, end + 1):
        top = tops[layer]
        if top < 0.0:
            top = 0.0
        if layer + 1 < n:
            bottom = tops[layer + 1]
        else:
            bottom = source_depth
        if bottom > source_depth:
            bottom = source_depth
        thk = bottom - top
        if thk <= eps:
            continue
        inv_v = 1.0 / velocities[layer]
        eta2 = inv_v * inv_v - p * p
        if eta2 < 1e-300:
            eta2 = 1e-300
        eta = math.sqrt(eta2)
        x += thk * p / eta
        t += thk * inv_v * inv_v / eta
    return x, t


@njit(cache=True)
def _direct_point(
    tops: np.ndarray,
    velocities: np.ndarray,
    v_source: float,
    source_depth: float,
    distance: float,
    eps: float,
):
    """First-arrival direct/upgoing branch from the source to a surface receiver.

    Solves the ray-parameter ``p`` whose summed horizontal distance equals the
    requested offset with a monotone bisection, then evaluates travel time and
    the source takeoff angle.
    """
    if source_depth <= eps:
        time_s = distance / velocities[0]
        angle = 180.0 if distance <= eps else 90.0
        return time_s, angle

    source_layer = _layer_index(tops, source_depth)

    # Largest velocity among crossed layers sets the smallest critical slowness.
    n = tops.shape[0]
    v_max = 0.0
    for layer in range(0, source_layer + 1):
        top = tops[layer]
        if top < 0.0:
            top = 0.0
        if layer + 1 < n:
            bottom = tops[layer + 1]
        else:
            bottom = source_depth
        if bottom > source_depth:
            bottom = source_depth
        if bottom - top > eps and velocities[layer] > v_max:
            v_max = velocities[layer]
    p_crit = 1.0 / v_max

    if distance <= eps:
        # Vertical ray: distance term vanishes, time is the sum of thk / v.
        _, time_s = _direct_x_t(tops, velocities, source_depth, 0.0, eps)
        return time_s, 180.0

    lo = 0.0
    hi = p_crit * (1.0 - eps)
    p = 0.0
    for _ in range(48):
        p = 0.5 * (lo + hi)
        x_mid, _t = _direct_x_t(tops, velocities, source_depth, p, eps)
        if x_mid < distance:
            lo = p
        else:
            hi = p
    p = 0.5 * (lo + hi)
    _x, time_s = _direct_x_t(tops, velocities, source_depth, p, eps)

    s = p * v_source
    if s < 0.0:
        s = 0.0
    if s > 1.0:
        s = 1.0
    emergence = math.degrees(math.asin(s))
    return time_s, 180.0 - emergence


@njit(cache=True)
def _leg_x_t(
    tops: np.ndarray,
    velocities: np.ndarray,
    z0: float,
    z1: float,
    refractor_layer: int,
    p: float,
    eps: float,
):
    """Slant-leg distance and time between two depths at fixed slowness ``p``.

    Layers at or below ``refractor_layer`` are skipped because the critically
    refracted wave runs along the top of the refractor rather than through it.
    Returns ``(x, t, valid)``; ``valid`` is ``False`` when ``p`` cannot
    propagate through one of the overlying layers.
    """
    if z1 < z0:
        z0, z1 = z1, z0
    if z1 - z0 <= eps:
        return 0.0, 0.0, True

    n = tops.shape[0]
    start = _layer_index(tops, z0)
    end = _layer_index(tops, z1)
    x = 0.0
    t = 0.0
    for layer in range(start, end + 1):
        top = tops[layer]
        if top < z0:
            top = z0
        if layer + 1 < n:
            bottom = tops[layer + 1]
        else:
            bottom = z1
        if bottom > z1:
            bottom = z1
        thk = bottom - top
        if thk <= eps:
            continue
        if layer >= refractor_layer:
            continue
        inv_v = 1.0 / velocities[layer]
        eta2 = inv_v * inv_v - p * p
        if eta2 <= 0.0:
            return 0.0, 0.0, False
        eta = math.sqrt(eta2)
        x += thk * p / eta
        t += thk * inv_v * inv_v / eta
    return x, t, True


@njit(cache=True)
def solve_one_point(
    tops: np.ndarray,
    velocities: np.ndarray,
    v_source: float,
    source_depth: float,
    distance: float,
    max_depth: float,
    eps: float,
):
    """First-arrival travel time and takeoff angle for one depth/distance pair.

    Starts from the direct branch, then tests every physically valid critically
    refracted head-wave branch and keeps whichever arrival is fastest.
    """
    best_time, best_angle = _direct_point(
        tops, velocities, v_source, source_depth, distance, eps
    )

    n = tops.shape[0]
    source_layer = _layer_index(tops, source_depth)

    source_on_top = abs(source_depth - tops[source_layer]) < 1.0e-9 and source_layer > 0
    start_refractor = source_layer if source_on_top else source_layer + 1

    two_dist = 2.0 * distance
    reach = two_dist if two_dist > max_depth else max_depth
    depth_limit = source_depth + reach + 50.0

    for refractor in range(start_refractor, n):
        interface = tops[refractor]
        if interface > depth_limit:
            break

        v_ref = velocities[refractor]
        fastest_above = 0.0
        for k in range(refractor):
            if velocities[k] > fastest_above:
                fastest_above = velocities[k]

        if source_on_top and refractor == source_layer:
            if v_ref <= fastest_above + 1.0e-10:
                continue
        else:
            threshold = fastest_above if fastest_above > v_source else v_source
            if v_ref <= threshold + 1.0e-10:
                continue

        p = 1.0 / v_ref
        x_legs, t_legs, ok = _leg_x_t(
            tops, velocities, 0.0, interface, refractor, p, eps
        )
        if ok and interface > source_depth:
            x_dn, t_dn, ok_dn = _leg_x_t(
                tops, velocities, source_depth, interface, refractor, p, eps
            )
            x_legs += x_dn
            t_legs += t_dn
            ok = ok and ok_dn
        if not ok:
            continue
        if distance < x_legs:
            continue

        head_time = t_legs + (distance - x_legs) * p
        if head_time < best_time:
            best_time = head_time
            s = p * v_source
            if s < 0.0:
                s = 0.0
            if s > 1.0:
                s = 1.0
            best_angle = math.degrees(math.asin(s))

    return best_time, best_angle


@njit(cache=True, parallel=True)
def build_table(
    tops: np.ndarray,
    velocities: np.ndarray,
    v_source_grid: np.ndarray,
    depth_grid: np.ndarray,
    distance_grid: np.ndarray,
    max_depth: float,
    eps: float,
):
    """Fill a ``(depth, distance)`` travel-time and takeoff-angle table.

    ``v_source_grid`` holds the source velocity for each entry of ``depth_grid``.
    The depth axis is parallelized with :func:`numba.prange`.  Returns two 2-D
    arrays ``(travel_time_s, takeoff_angle_deg)``.
    """
    nd = depth_grid.shape[0]
    nr = distance_grid.shape[0]
    tt = np.empty((nd, nr), dtype=np.float64)
    ta = np.empty((nd, nr), dtype=np.float64)
    for i in prange(nd):
        z = depth_grid[i]
        vs = v_source_grid[i]
        for j in range(nr):
            t, a = solve_one_point(
                tops, velocities, vs, z, distance_grid[j], max_depth, eps
            )
            tt[i, j] = t
            ta[i, j] = a
    return tt, ta
