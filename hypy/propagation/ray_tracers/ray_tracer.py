from typing import List
from ..propagation_model import PropagationModel


class RayTracer(PropagationModel):
    """
    Unified RayTracer interface to select backend ('pyrocko' or 'taup').
    """

    def __init__(self, model_path: str, engine: str = "pyrocko"):
        self.tracer: PropagationModel
        if engine == "pyrocko":
            from .pyrocko import PyrockoRayTracer

            self.tracer = PyrockoRayTracer(model_path)
        elif engine == "taup":
            from .taup import TauPRayTracer

            self.tracer = TauPRayTracer(model_path)
        else:
            raise ValueError(f"Unknown backend: {engine}. Choose 'pyrocko' or 'taup'.")

    def get_arrival(self, source_depth: float, distance: float, phase: str):
        return self.tracer.get_arrival(source_depth, distance, phase)

    def compute_travel_parameters(
        self,
        source_lon: float,
        source_lat: float,
        source_depth_km: float,
        receiver_lon: float,
        receiver_lat: float,
        receiver_elev_m: float,
        phase: str,
    ):
        return self.tracer.compute_travel_parameters(
            source_lon,
            source_lat,
            source_depth_km,
            receiver_lon,
            receiver_lat,
            receiver_elev_m,
            phase,
        )

    def compute_bulk_arrival_parameters(
        self,
        source_lon: float,
        source_lat: float,
        source_depth_km: float,
        receiver_lons: List[float],
        receiver_lats: List[float],
        receiver_elevs_m: List[float],
        phase: str,
    ):
        return self.tracer.compute_bulk_arrival_parameters(
            source_lon,
            source_lat,
            source_depth_km,
            receiver_lons,
            receiver_lats,
            receiver_elevs_m,
            phase,
        )
