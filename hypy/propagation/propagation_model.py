from pathlib import Path
from typing import List


class PropagationModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = Path(model_path).stem
        self.model = None

    def get_arrival(self, source_depth, distance, phase):
        # Placeholder for child classes to implement
        # arrival time and takeoff angle retrieval
        raise NotImplementedError("This method should be overridden by subclasses")

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
        # Placeholder for child classes to implement
        # arrival parameter calculation
        raise NotImplementedError("This method should be overridden by subclasses")

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
        """
        Get arrival parameters for all phases between source and multiple receivers.
        """
        raise NotImplementedError("This method should be overridden by subclasses")
