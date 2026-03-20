from typing import Optional

from pathlib import Path


class PropagationModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = Path(model_path).stem

    def get_arrival(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_elevation_m: float,
        phase: str,
        max_distance_km: Optional[float] = None,
    ):
        # Placeholder for child classes to implement
        # arrival time and takeoff angle retrieval
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_spatial_slowness(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_elevation_m: float,
        phase: str,
        delta=0.01,
    ):
        # Placeholder for child classes to implement
        # spatial slowness retrieval
        raise NotImplementedError("This method should be overridden by subclasses")
