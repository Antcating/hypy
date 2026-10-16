"""Tests for pyekfmm-based volumetric travel-time backends."""

import os
from pathlib import Path

import numpy as np
import obspy  # type: ignore[import-untyped]
import pytest

from hypy.locator import EventLocator
from hypy.model.layered.tvel import TvelModel
from hypy.propagation.volumetric.grid.pyekfmm import FmmTravelTimeGrid3D


def _iasp91_tvel_path() -> str:
    obspy_path = Path(obspy.__file__).parents[0].as_posix()
    return os.path.join(obspy_path, "taup/data/iasp91.tvel")


def _tvel_fmm_engine(**overrides):
    kwargs = dict(
        model=_iasp91_tvel_path(),
        phases=["P", "S"],
        max_distance_km=200.0,
        distance_step_km=1.0,
        max_depth_km=50.0,
        depth_step_km=1.0,
    )
    kwargs.update(overrides)
    return FmmTravelTimeGrid3D(**kwargs)


def test_pyekfmm_arrival_from_taup_nll_grid():
    """TVEL-backed pyekfmm should return valid first-arrival travel times."""
    tracer = _tvel_fmm_engine()

    arrival_p = tracer.get_arrival(
        source_latitude=10.0,
        source_longitude=10.0,
        source_depth_km=2.0,
        receiver_latitude=10.1,
        receiver_longitude=10.1,
        receiver_elevation_m=0.0,
        phase="P",
    )
    arrival_s = tracer.get_arrival(
        source_latitude=10.0,
        source_longitude=10.0,
        source_depth_km=2.0,
        receiver_latitude=10.1,
        receiver_longitude=10.1,
        receiver_elevation_m=0.0,
        phase="S",
    )

    arc_distance_km_p, travel_time_p, takeoff_angle_p, azimuth_p = arrival_p
    arc_distance_km_s, travel_time_s, takeoff_angle_s, azimuth_s = arrival_s

    assert arc_distance_km_p > 0.0
    assert arc_distance_km_s > 0.0
    assert travel_time_p is not None
    assert travel_time_s is not None
    assert travel_time_p > 0.0
    assert travel_time_s > 0.0
    assert travel_time_s > travel_time_p
    assert takeoff_angle_p is None
    assert takeoff_angle_s is None
    assert 0.0 <= azimuth_p < 360.0
    assert 0.0 <= azimuth_s < 360.0


def test_fmm3d_matches_locator_phase_and_derivative_contract():
    """The 3D engine advertises the flags and lowercase phases the locator
    requires."""
    tracer = _tvel_fmm_engine()
    assert tracer.dim == 3
    assert tracer.phases == ("p", "s")
    assert "derivative.numerical" in tracer.feature_flags
    assert "model.grid.3d" in tracer.velocity_model.feature_flags

    locator = EventLocator(tracer, phases=("P", "S"), derivative_method="numerical")
    assert locator.propagation_backend is tracer


def test_fmm3d_rejects_layered_tvel_instance():
    """A 1D TvelModel instance is not a volumetric grid provider."""
    model = TvelModel(_iasp91_tvel_path())
    with pytest.raises(ValueError, match="missing required features"):
        FmmTravelTimeGrid3D(
            model=model,
            phases=["P"],
            max_distance_km=20.0,
            distance_step_km=2.0,
            max_depth_km=10.0,
            depth_step_km=1.0,
        )


def test_pyekfmm_compute_travel_time_params():
    """Numerical derivatives are finite for a TVEL-backed 3D engine."""
    tracer = _tvel_fmm_engine(
        max_distance_km=30.0,
        distance_step_km=5.0,
        max_depth_km=20.0,
        depth_step_km=5.0,
    )
    arc, travel_time, derivs = tracer.compute_travel_time_params(
        source_latitude=10.0,
        source_longitude=10.0,
        source_depth_km=5.0,
        receiver_latitude=10.05,
        receiver_longitude=10.05,
        receiver_elevation_m=0.0,
        phase="P",
        delta=1.0,
        derivative_method="numerical",
    )
    assert arc > 0.0
    assert travel_time is not None and travel_time > 0.0
    assert all(d is not None and np.isfinite(d) for d in derivs)


def test_event_locator_rejects_analytical_3d():
    """EventLocator rejects analytical derivatives for the 3D FMM backend."""
    tracer = _tvel_fmm_engine(
        max_distance_km=20.0,
        distance_step_km=5.0,
        max_depth_km=10.0,
        depth_step_km=5.0,
    )
    with pytest.raises(ValueError, match="Unsupported derivative method"):
        EventLocator(tracer, phases=("P", "S"), derivative_method="analytical")
