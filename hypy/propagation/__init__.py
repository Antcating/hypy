from pathlib import Path
from typing import Optional


class PropagationModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = Path(model_path).stem

    @property
    def dim(self) -> int:
        """Return the dimensionality of the propagation model.

        This is a placeholder method and should be overridden by subclasses
        to return the actual dimensionality of the model (e.g., 1 for 1D, 3 for 3D).

        Returns
        -------
        int
            The dimensionality of the propagation model.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

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
    ) -> tuple[float, Optional[float], Optional[float], float]:
        # Placeholder for child classes to implement
        # arrival time and takeoff angle retrieval
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_spatial_slowness(self, *args, **kwargs):
        # Placeholder for child classes to implement
        # spatial slowness retrieval
        raise NotImplementedError("This method should be overridden by subclasses")
