"""Layered ray tracing module for seismic wave propagation."""

from hypy.model import VelocityModel
from hypy.propagation.layered import LayeredPropagationEngine


class LayeredRayTracer(LayeredPropagationEngine):
    """
    Ray tracer for layered Earth models.

    Supports multiple backends (e.g., Pyrocko, TauP), but with
    unified interface for computing ray parameters like travel time and takeoff angle
    for given source-receiver configurations and seismic phases.
    """

    def __init__(self, model: VelocityModel):
        super().__init__(model)
