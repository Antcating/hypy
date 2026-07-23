"""Tests for layered ray-tracer backends."""

import os

import obspy
import pytest

from hypy.model.layered.tvel import TvelModel
from hypy.propagation.layered.ray_tracer import LayeredRayTracer
from hypy.propagation.layered.ray_tracer.pyrocko import PyrockoRayTracer
from hypy.propagation.layered.ray_tracer.taup import TauPRayTracer


@pytest.fixture(params=["taup", "pyrocko"])
def tracer(request):
    """Build a layered ray-tracer backend for each supported engine."""
    obspy_path = os.path.dirname(obspy.__file__)
    model_path = os.path.join(obspy_path, "taup", "data", "iasp91.tvel")
    model = TvelModel(model_path)
    if request.param == "taup":
        return TauPRayTracer(model=model)
    elif request.param == "pyrocko":
        return PyrockoRayTracer(model=model)


def test_get_arrival_1d(tracer: LayeredRayTracer):
    """1D ray-tracer arrivals should match reference values."""
    depth = 50.0
    distance = 500.0
    time_p, angle_p = tracer.get_arrival_1d(distance, depth, "P")
    time_s, angle_s = tracer.get_arrival_1d(distance, depth, "S")

    assert isinstance(time_p, float)
    assert pytest.approx(time_p, rel=1e-3) == 65.54952023370768
    assert isinstance(angle_p, float)
    assert pytest.approx(angle_p, rel=1e-1) == 89.6292410507832
    assert isinstance(time_s, float)
    assert pytest.approx(time_s, rel=1e-3) == 117.19924749280987
    assert isinstance(angle_s, float)
    assert pytest.approx(angle_s, rel=1e-1) == 88.83770931419198


def test_get_arrival_1d_invalid_phase(tracer: LayeredRayTracer):
    """Invalid 1D phases should raise a validation error."""
    depth = 50.0
    distance = 500.0
    with pytest.raises(ValueError):
        tracer.get_arrival_1d(depth, distance, "X")


def test_get_arrival_1d_out_of_bounds(tracer: LayeredRayTracer):
    """Out-of-range 1D depths should raise a validation error."""
    depth = -50.0
    distance = 500.0
    with pytest.raises(ValueError):
        tracer.get_arrival_1d(distance, depth, "P")
    with pytest.raises(ValueError):
        tracer.get_arrival_1d(distance, depth, "S")


def test_get_arrival_1d_negative_distance(tracer: LayeredRayTracer):
    """Negative distances should mirror positive-distance arrivals."""
    depth = 50.0
    distance = -100.0
    assert tracer.get_arrival_1d(distance, depth, "P") == tracer.get_arrival_1d(
        -distance, depth, "P"
    )
    assert tracer.get_arrival_1d(distance, depth, "S") == tracer.get_arrival_1d(
        -distance, depth, "S"
    )


def test_get_arrival(tracer: LayeredRayTracer):
    """2D ray-tracer arrivals should match the geodesic reference values."""
    source_latitude = 0.0
    source_longitude = 0.0
    source_depth_km = 50.0
    receiver_latitude = 0.0
    receiver_longitude = 4.5  # ~500 km at the equator
    receiver_elevation_m = 0.0
    arc_distance_km_p, time_p, takeoff_angle_p, azimuth_deg_p = tracer.get_arrival(
        source_latitude,
        source_longitude,
        source_depth_km,
        receiver_latitude,
        receiver_longitude,
        receiver_elevation_m,
        "P",
    )

    assert isinstance(time_p, float)
    assert pytest.approx(time_p, rel=1e-3) == 65.54952023370768
    assert isinstance(takeoff_angle_p, float)
    assert pytest.approx(takeoff_angle_p, rel=1e-1) == 89.6292410507832
    assert isinstance(azimuth_deg_p, float)
    assert pytest.approx(azimuth_deg_p, rel=1e-1) == 90.0
    assert isinstance(arc_distance_km_p, float)
    assert pytest.approx(arc_distance_km_p, rel=1e-3) == 500.0

    arc_distance_km_s, time_s, takeoff_angle_s, azimuth_deg_s = tracer.get_arrival(
        source_latitude,
        source_longitude,
        source_depth_km,
        receiver_latitude,
        receiver_longitude,
        receiver_elevation_m,
        "S",
    )

    assert isinstance(time_s, float)
    assert pytest.approx(time_s, rel=1e-3) == 117.19924749280987
    assert isinstance(takeoff_angle_s, float)
    assert pytest.approx(takeoff_angle_s, rel=1e-1) == 88.83770931419198
    assert isinstance(azimuth_deg_s, float)
    assert pytest.approx(azimuth_deg_s, rel=1e-1) == 90.0
    assert isinstance(arc_distance_km_s, float)
    assert pytest.approx(arc_distance_km_s, rel=1e-3) == 500.0


def test_get_arrival_invalid_phase(tracer: LayeredRayTracer):
    """Invalid phases should raise a validation error for 2D arrivals."""
    source_latitude = 0.0
    source_longitude = 0.0
    source_depth_km = 50.0
    receiver_latitude = 0.0
    receiver_longitude = 4.5
    receiver_elevation_m = 0.0
    with pytest.raises(ValueError):
        tracer.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "X",
        )


def test_get_arrival_out_of_bounds(tracer: LayeredRayTracer):
    """Out-of-range depths should raise a validation error for arrivals."""
    source_latitude = 0.0
    source_longitude = 0.0
    source_depth_km = -50.0
    receiver_latitude = 0.0
    receiver_longitude = 4.5
    receiver_elevation_m = 0.0
    with pytest.raises(ValueError):
        tracer.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "P",
        )
    with pytest.raises(ValueError):
        tracer.get_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            receiver_latitude,
            receiver_longitude,
            receiver_elevation_m,
            "S",
        )


def test_get_arrival_max_distance_exceeded(tracer: LayeredRayTracer):
    """Exceeding the max distance should suppress travel-time computation."""
    source_latitude = 0.0
    source_longitude = 0.0
    source_depth_km = 50.0
    receiver_latitude = 0.0
    receiver_longitude = 10.0
    receiver_elevation_m = 0.0
    max_distance_km = 500.0
    arc_distance_km_p, time_p, takeoff_angle_p, azimuth_deg_p = tracer.get_arrival(
        source_latitude,
        source_longitude,
        source_depth_km,
        receiver_latitude,
        receiver_longitude,
        receiver_elevation_m,
        "P",
        max_distance_km=max_distance_km,
    )
    assert time_p is None
    assert takeoff_angle_p is None
    assert isinstance(azimuth_deg_p, float)
    assert pytest.approx(azimuth_deg_p, rel=1e-1) == 90.0
    assert isinstance(arc_distance_km_p, float)
    assert pytest.approx(arc_distance_km_p, rel=1e-3) == 1111.9
    arc_distance_km_s, time_s, takeoff_angle_s, azimuth_deg_s = tracer.get_arrival(
        source_latitude,
        source_longitude,
        source_depth_km,
        receiver_latitude,
        receiver_longitude,
        receiver_elevation_m,
        "S",
        max_distance_km=max_distance_km,
    )

    assert time_s is None
    assert takeoff_angle_s is None
    assert isinstance(azimuth_deg_s, float)
    assert pytest.approx(azimuth_deg_s, rel=1e-1) == 90.0
    assert isinstance(arc_distance_km_s, float)
    assert pytest.approx(arc_distance_km_s, rel=1e-3) == 1111.9
