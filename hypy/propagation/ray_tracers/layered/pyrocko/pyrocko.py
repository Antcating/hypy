from typing import List

try:
    from pyrocko import cake  # type: ignore[import-untyped]
    from pyrocko.cake import Ray  # type: ignore[import-untyped]
except ImportError:
    raise ImportError(
        "Pyrocko is not installed or not available. Please install Pyrocko to use the pyrocko engine."
    )
from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]

from hypy.propagation import PropagationModel


class PyrockoRayTracer(PropagationModel):
    """
    Ray tracer using Pyrocko's cake module for seismic ray tracing.
    """

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model: cake.LayeredModel = cake.load_model(model_path)

    def get_arrival(self, source_depth: float, distance: float, phase: str):
        """
        Get the arrival time for a given source depth, distance, and seismic phase.
        """
        if phase.lower() == "p":
            phases = ["p", "P"]
        elif phase.lower() == "s":
            phases = ["s", "S"]
        else:
            raise ValueError(
                f"Unsupported phase: {phase}. Supported phases are 'P' and 'S'."
            )

        # Convert distance from km to degrees
        try:
            arrivals: List[Ray] = self.model.arrivals(
                zstart=source_depth * 1000,
                zstop=0,
                distances=[kilometer2degrees(distance)],
                phases=phases,
            )
        except Exception as e:
            raise ValueError(
                f"Error computing arrivals for phase {phase} at distance {distance} and depth {source_depth}: {e}"
            )
        if not arrivals:
            raise ValueError(
                f"No arrivals found for phase {phase} at distance {distance} and depth {source_depth}."
            )

        return arrivals[0].t, arrivals[0].takeoff_angle()
