"""1D ray tracer based on Pyrocko's cake module.

Defines :class:`PyrockoRayTracer`, which implements the
:class:`~hypy.propagation.layered.ray_tracer.LayeredRayTracer` interface
using Pyrocko for travel time and takeoff angle calculations. Importing
this module requires Pyrocko to be installed.
"""

from typing import Any, List

from hypy.model import VelocityModel

try:
    from pyrocko import cake  # type: ignore[import-untyped]
    from pyrocko.cake import Ray  # type: ignore[import-untyped]
except ImportError:
    raise ImportError(
        "Pyrocko is not installed or not available."
        " Please install Pyrocko to use the pyrocko engine."
    )
from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]

from hypy.propagation.layered.ray_tracer import LayeredRayTracer


class PyrockoRayTracer(LayeredRayTracer):
    """Seismic Ray Tracer using pyrocko's cake toolkit.

    It provides methods to compute ray parametes for seismic phases
    and given source-receiver configurations and a 1D velocity model.

    Parameters
    ----------
    LayeredRayTracer : LayeredPropagationModel
        Base class for layered ray tracing.

    Additional keyword arguments are forwarded to ``LayeredRayTracer``.  Use
    ``arrival_cache_size`` and ``arrival_cache_precision_km`` there to configure
    the shared layered-arrival cache
    """

    feature_flags = frozenset(
        {
            "propagation.layered",
            "propagation.ray",
            "derivative.analytical",
            "derivative.numerical",
        }
    )
    required_model_features = frozenset({"model.profile.1d"})
    name = "pyrocko"

    def __init__(self, model: VelocityModel, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.model: cake.LayeredModel = cake.load_model(model.model_path.as_posix())

    def _compute_arrival_1d(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ) -> tuple[float, float]:
        """Return the first Pyrocko arrival time and takeoff angle.

        For 1D layered models these depend only on source depth and arc
        distance, not on the specific geographic locations.

        Raises
        ------
        ValueError
            If cake errors out, or finds no arrival for this geometry.
        """
        try:
            arrivals: List[Ray] = self.model.arrivals(
                zstart=source_depth_km * 1000,
                zstop=0,
                distances=[kilometer2degrees(arc_distance_km)],
                phases=[phase, phase.upper()],
            )
        except Exception as e:
            raise ValueError(
                f"Error computing arrivals for phase {phase}"
                f" at distance {arc_distance_km} and depth {source_depth_km}: {e}"
            )
        if len(arrivals) == 0:
            raise ValueError(
                f"No arrivals found for phase {phase} "
                f"at distance {arc_distance_km} and depth {source_depth_km}."
            )
        return float(arrivals[0].t), float(arrivals[0].takeoff_angle())
