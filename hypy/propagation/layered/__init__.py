from typing import Optional

from obspy.taup import TauPyModel  # type: ignore[import-untyped]

from hypy.logger import logger
from hypy.propagation import PropagationModel
from hypy.utils.geo import gps2dist_azimuth_nb


class LayeredPropagationModel(PropagationModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.velocity_model = TauPyModel(self.model_name)

    def compute_arc_params(
        self,
        source_latitude: float,
        source_longitude: float,
        receiver_latitude: float,
        receiver_longitude: float,
    ) -> tuple[float, float, float]:
        """
        Compute the arc distance between source and receiver in kilometers.
        """
        dist_m, azimuth, back_azimuth = gps2dist_azimuth_nb(
            source_latitude,
            source_longitude,
            receiver_latitude,
            receiver_longitude,
            R=self.velocity_model.model.radius_of_planet * 1000.0,
        )
        return dist_m / 1000.0, azimuth, back_azimuth
