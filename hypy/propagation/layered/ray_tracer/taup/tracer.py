"""1D TauP ray tracer based on ObsPy's TauP module.

Defines :class:`TauPRayTracer`, which implements the
:class:`~hypy.propagation.layered.ray_tracer.LayeredRayTracer` interface
using TauP for travel time and takeoff angle calculations.
"""

from typing import Any

from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]
from obspy.taup import TauPyModel  # type: ignore[import-untyped]
from obspy.taup.tau import Arrival  # type: ignore[import-untyped]

from hypy.model import VelocityModel
from hypy.propagation.layered.ray_tracer import LayeredRayTracer


class TauPRayTracer(LayeredRayTracer):
    """Seismic Ray Tracer using obspy's TauP toolkit.

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
    name = "taup"

    def __init__(self, model: VelocityModel, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.model: TauPyModel = TauPyModel(model.model_name)

    def _compute_arrival_1d(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ) -> tuple[float, float]:
        """Return the first TauP arrival time and takeoff angle.

        For 1D layered models these depend only on source depth and arc
        distance, not on the specific geographic locations.

        Raises
        ------
        ValueError
            If TauP errors out, or finds no arrival for this geometry.
        """
        try:
            phase_group = "ttp" if phase == "p" else "tts"
            arrivals: list[Arrival] = self.model.get_travel_times(
                source_depth_in_km=source_depth_km,
                distance_in_degree=kilometer2degrees(arc_distance_km),
                phase_list=[phase_group],
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
        return float(arrivals[0].time), float(arrivals[0].takeoff_angle)
