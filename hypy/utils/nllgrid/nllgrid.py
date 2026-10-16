"""Utility functions for working with NLLGrids and TauP models."""

from pathlib import Path

import numpy as np
from nllgrid import NLLGrid  # type: ignore[import-untyped]
from obspy.taup import TauPyModel  # type: ignore[import-untyped]
from obspy.taup.taup_create import build_taup_model  # type: ignore[import-untyped]


def _load_taup_model_from_tvel(model_tvel: str | Path) -> TauPyModel:
    tvel_path = Path(model_tvel).expanduser().resolve()
    model_name = tvel_path.stem
    try:
        return TauPyModel(model_name)
    except FileNotFoundError:
        build_taup_model(tvel_path.as_posix())
        return TauPyModel(model_name)


def create_nll_grid_from_tvel(
    model_tvel: str | Path,
    phase: str = "P",
    nx: int = 21,
    ny: int = 41,
    nz: int = 11,
    dx: float = 0.5,
    dy: float = 0.5,
    dz: float = 0.5,
    x_orig: float = 0.0,
    y_orig: float = 0.0,
    z_orig: float = 0.0,
    orig_lat: float = 40.63,
    orig_lon: float = 15.80,
    proj_name: str = "LAMBERT",
    first_std_paral: float = 38.0,
    second_std_paral: float = 42.0,
    proj_ellipsoid: str = "WGS-84",
) -> NLLGrid:
    """Build a 3D NLL velocity grid from a 1D TauP-compatible TVEL model.

    The 1D velocity profile is sampled along depth (z) and replicated
    along x/y. Depths above sea level are clamped to 0 km for TauP
    queries.
    """
    model = _load_taup_model_from_tvel(model_tvel)
    if model.model.s_mod is None:
        raise ValueError("TauP velocity model 's_mod' is not initialized.")

    v_mod = model.model.s_mod.v_mod
    phase_code = phase.lower()[-1]

    z_km = z_orig + np.arange(nz, dtype=float) * dz
    z_query_km = np.maximum(0.0, z_km)

    velocities = np.array(
        [v_mod.evaluate_below(depth_km, phase_code)[0] for depth_km in z_query_km],
        dtype=np.float32,
    )

    velocity_grid = np.broadcast_to(
        velocities[np.newaxis, np.newaxis, :], (nx, ny, nz)
    ).copy()

    grd = NLLGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        x_orig=x_orig,
        y_orig=y_orig,
        z_orig=z_orig,
        dx=dx,
        dy=dy,
        dz=dz,
    )
    grd.array = velocity_grid
    grd.type = "VELOCITY"

    grd.orig_lat = orig_lat
    grd.orig_lon = orig_lon
    grd.proj_name = proj_name
    grd.first_std_paral = first_std_paral
    grd.second_std_paral = second_std_paral
    grd.proj_ellipsoid = proj_ellipsoid

    return grd
