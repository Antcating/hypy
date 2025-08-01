from typing import List
from pyrocko import cake
from pyrocko.cake import Ray
from hypy.propagation import PropagationModel

from obspy.geodetics import kilometer2degrees

class PyrockoRayTracer(PropagationModel):
    """
    Ray tracer using Pyrocko's cake module for seismic ray tracing.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = cake.load_model(model_name)

    def get_arrival(self, source_depth: float, distance: float, phase: str):
        """
        Get the arrival time for a given source depth, distance, and seismic phase.
        """
        if phase.lower() == 'p':
            phases = ["p", "P"]
        elif phase.lower() == 's':
            phases = ["s", "S"]
        else:
            raise ValueError(f"Unsupported phase: {phase}. Supported phases are 'P' and 'S'.")
        
         # Convert distance from km to degrees
        arrivals: List[Ray] = self.model.arrivals(
            zstart=source_depth * 1000,
            zstop=0,
            distances=[kilometer2degrees(distance)],
            phases=phases,
        )
        if not arrivals:
            raise ValueError(f"No arrivals found for phase {phase} at distance {distance} and depth {source_depth}.")
        

        return arrivals[0].t, arrivals[0].takeoff_angle()