"""Tests for layered ray-tracer backends."""

import os

import obspy
import pytest

from hypy.model.layered.tvel import TvelModel
from hypy.propagation.layered import LayeredPropagationEngine
from hypy.propagation.layered.grid.fmm import FmmTravelTimeGrid1D
from hypy.propagation.layered.grid.tau import TauTravelTimeGrid
from hypy.propagation.layered.ray_tracer.hypo import HypoRayTracer
from hypy.propagation.layered.ray_tracer.pyrocko import PyrockoRayTracer
from hypy.propagation.layered.ray_tracer.taup import TauPRayTracer

# Create all the propagation engines
# - Ray-tracer backends:
#   - TauPRayTracer
#   - PyrockoRayTracer
#   - HypoRayTracer
# - Grid propagation engines:
#   - TauTravelTimeGrid with Pyrocko backend
#   - TauTravelTimeGrid with TauP backend
#   - FmmTravelTimeGrid1D


@pytest.fixture(
    params=[
        ("ray_tracer", "taup"),
        ("ray_tracer", "pyrocko"),
        ("ray_tracer", "hypo"),
        ("grid", "tau"),
        ("grid", "pyrocko"),
        ("grid", "hypo"),
        ("grid", "fmm"),
    ],
    ids=[
        "ray_tracer-taup",
        "ray_tracer-pyrocko",
        "ray_tracer-hypo",
        "grid-tau",
        "grid-pyrocko",
        "grid-hypo",
        "grid-fmm",
    ],
)
def propagation_engine(request):
    mode, engine = request.param
    obspy_path = os.path.dirname(obspy.__file__)
    model_path = os.path.join(obspy_path, "taup", "data", "iasp91.tvel")
    model = TvelModel(model_path)
    if mode == "ray_tracer":
        if engine == "taup":
            return TauPRayTracer(model=model)
        elif engine == "pyrocko":
            return PyrockoRayTracer(model=model)
        elif engine == "hypo":
            return HypoRayTracer(model=model)
    elif mode == "grid":
        if engine == "tau":
            return TauTravelTimeGrid(
                model=model,
                tracer=TauPRayTracer(model=model),
                phases=["P", "S"],
                max_depth_km=20,
                max_distance_km=80,
                depth_step_km=1,
                distance_step_km=1,
            )
        elif engine == "pyrocko":
            return TauTravelTimeGrid(
                model=model,
                tracer=PyrockoRayTracer(model=model),
                phases=["P", "S"],
                max_depth_km=20,
                max_distance_km=80,
                depth_step_km=1,
                distance_step_km=1,
            )
        elif engine == "hypo":
            return TauTravelTimeGrid(
                model=model,
                tracer=HypoRayTracer(model=model),
                phases=["P", "S"],
                max_depth_km=20,
                max_distance_km=80,
                depth_step_km=1,
                distance_step_km=1,
            )
        elif engine == "fmm":
            return FmmTravelTimeGrid1D(
                model=model,
                phases=["P", "S"],
                max_depth_km=20,
                max_distance_km=80,
                depth_step_km=1,
                distance_step_km=1,
            )


def test_arrival_times(propagation_engine: LayeredPropagationEngine):
    """Test arrival times for a given depth and distance."""
    depth_km = 10.0
    distance_km = 70.0
    time_p, _ = propagation_engine.get_arrival_1d(distance_km, depth_km, "P")
    time_s, _ = propagation_engine.get_arrival_1d(distance_km, depth_km, "S")
    assert isinstance(time_p, float)
    assert pytest.approx(time_p, rel=1e-2) == 12.18
    assert isinstance(time_s, float)
    assert pytest.approx(time_s, rel=1e-2) == 21.02


def test_takeoff_angles(propagation_engine: LayeredPropagationEngine):
    """Test takeoff angles for a given depth and distance."""

    if isinstance(propagation_engine, FmmTravelTimeGrid1D):
        return
    depth_km = 10.0
    distance_km = 70.0
    _, angle_p = propagation_engine.get_arrival_1d(distance_km, depth_km, "P")
    _, angle_s = propagation_engine.get_arrival_1d(distance_km, depth_km, "S")
    assert isinstance(angle_p, float)
    assert pytest.approx(angle_p, rel=1e-2) == 97.82
    assert isinstance(angle_s, float)
    assert pytest.approx(angle_s, rel=1e-2) == 97.82


def test_get_arrival_1d_invalid_phase(
    propagation_engine: LayeredPropagationEngine,
):
    """Invalid 1D phases should raise a validation error."""
    depth = 50.0
    distance = 500.0
    with pytest.raises(ValueError):
        propagation_engine.get_arrival_1d(distance, depth, "X")


def test_get_arrival_1d_out_of_bounds(
    propagation_engine: LayeredPropagationEngine,
):
    """Out-of-range 1D depths should raise a validation error."""
    depth_km = -20.0
    distance_km = 70.0
    with pytest.raises(ValueError):
        propagation_engine.get_arrival_1d(distance_km, depth_km, "P")
    with pytest.raises(ValueError):
        propagation_engine.get_arrival_1d(distance_km, depth_km, "S")


def test_get_arrival_1d_negative_distance(
    propagation_engine: LayeredPropagationEngine,
):
    """Negative distances should mirror positive-distance arrivals."""
    depth_km = 10.0
    distance_km = -10.0
    assert propagation_engine.get_arrival_1d(
        distance_km, depth_km, "P"
    ) == propagation_engine.get_arrival_1d(-distance_km, depth_km, "P")
    assert propagation_engine.get_arrival_1d(
        distance_km, depth_km, "S"
    ) == propagation_engine.get_arrival_1d(-distance_km, depth_km, "S")


def test_get_arrival_times(propagation_engine: LayeredPropagationEngine):
    """2D ray-tracer arrivals should match the geodesic reference values."""
    source_latitude = 0.0
    source_longitude = 0.0
    source_depth_km = 10.0
    receiver_latitude = 0.0
    receiver_longitude = 0.1  # ~10 km
    receiver_elevation_m = 0.0
    arc_distance_km_p, time_p, takeoff_angle_p, azimuth_deg_p = (
        propagation_engine.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "P",
        )
    )

    assert isinstance(time_p, float)
    assert pytest.approx(time_p, rel=1e-1) == 2.57
    assert isinstance(azimuth_deg_p, float)
    assert pytest.approx(azimuth_deg_p, rel=1e-2) == 90.0
    assert isinstance(arc_distance_km_p, float)
    assert pytest.approx(arc_distance_km_p, rel=1e-2) == 11.11

    arc_distance_km_s, time_s, takeoff_angle_s, azimuth_deg_s = (
        propagation_engine.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "S",
        )
    )

    assert isinstance(time_s, float)
    assert pytest.approx(time_s, rel=1e-1) == 4.44
    assert isinstance(azimuth_deg_s, float)
    assert pytest.approx(azimuth_deg_s, rel=1e-2) == 90.0
    assert isinstance(arc_distance_km_s, float)
    assert pytest.approx(arc_distance_km_s, rel=1e-2) == 11.11


def test_get_arrival_invalid_phase(propagation_engine: LayeredPropagationEngine):
    """Invalid phases should raise a validation error for 2D arrivals."""
    source_latitude = 0.0
    source_longitude = 0.0
    source_depth_km = 10.0
    receiver_latitude = 0.0
    receiver_longitude = 0.1
    receiver_elevation_m = 0.0
    with pytest.raises(ValueError):
        propagation_engine.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "X",
        )


def test_get_arrival_out_of_bounds(propagation_engine: LayeredPropagationEngine):
    """Out-of-range depths should raise a validation error for arrivals."""
    source_latitude = 0.0
    source_longitude = 0.0
    source_depth_km = -50.0
    receiver_latitude = 0.0
    receiver_longitude = 4.5
    receiver_elevation_m = 0.0
    with pytest.raises(ValueError):
        propagation_engine.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "P",
        )
    with pytest.raises(ValueError):
        propagation_engine.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "S",
        )


def test_get_arrival_max_distance_exceeded(
    propagation_engine: LayeredPropagationEngine,
):
    """Exceeding the max distance should suppress travel-time computation."""
    source_latitude = 0.0
    source_longitude = 0.0
    source_depth_km = 10.0
    receiver_latitude = 0.0
    receiver_longitude = 10.0
    receiver_elevation_m = 0.0
    max_distance_km = 500.0
    arc_distance_km_p, time_p, takeoff_angle_p, azimuth_deg_p = (
        propagation_engine.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "P",
            max_distance_km=max_distance_km,
        )
    )
    assert time_p is None
    assert takeoff_angle_p is None
    assert isinstance(azimuth_deg_p, float)
    assert pytest.approx(azimuth_deg_p, rel=1e-1) == 90.0
    assert isinstance(arc_distance_km_p, float)
    assert pytest.approx(arc_distance_km_p, rel=1e-3) == 1111.9
    arc_distance_km_s, time_s, takeoff_angle_s, azimuth_deg_s = (
        propagation_engine.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "S",
            max_distance_km=max_distance_km,
        )
    )

    assert time_s is None
    assert takeoff_angle_s is None
    assert isinstance(azimuth_deg_s, float)
    assert pytest.approx(azimuth_deg_s, rel=1e-1) == 90.0
    assert isinstance(arc_distance_km_s, float)
    assert pytest.approx(arc_distance_km_s, rel=1e-3) == 1111.9
