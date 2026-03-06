from typing import List, Union

import numpy as np

from .config import LocatorConfig
from .data import DataHandler
from .types import (
    Arrival,
    Catalog,
    ConfidenceEllipsoid,
    ErrorResults,
    Event,
    Origin,
    OriginUncertainty,
    QuantityError,
)


class OriginFinalizer:
    """
    Converts located event data into ObsPy Origin objects with uncertainty information.

    All dependencies are passed explicitly. Does not own event/station data.
    """

    def __init__(self, config: LocatorConfig, data: DataHandler) -> None:
        self.config = config
        self.data = data

    def finalize_origins(
        self, error_results: ErrorResults
    ) -> Union[List[Event], Catalog]:
        """
        Attach final Origin objects to events:
        - Convert stored relative origin times (t0_rel) back to absolute by adding time_refs.
        - Locations/depths are left as-is (km for depth, converted to meters for ObsPy).
        - Arrival residuals are computed in the same sign convention used in inversion:
          residual = (observed_travel_time - theoretical_travel_time).
        """
        config = self.config
        data = self.data
        scale95 = 2.4

        for event_idx, event in enumerate(data.events):
            # If invalid - no origins are added
            if data.invalid_events[event_idx]:
                continue

            origin = Origin()
            origin.latitude = data.locations[event_idx, 0]
            origin.longitude = data.locations[event_idx, 1]
            origin.depth = data.locations[event_idx, 2] * 1000.0  # km -> m

            # Convert RELATIVE t0 back to ABSOLUTE for output
            t0_abs = float(data.origin_times[event_idx]) + float(
                data.time_refs[event_idx]
            )
            origin.time = t0_abs

            # Uncertainties / ellipsoid
            s1, s2, s3 = error_results.principal_std[event_idx]
            evecs = error_results.principal_axes[event_idx]
            ERZ_1s, ERH_1s = error_results.erz_erh[event_idx]

            v = evecs[:, 0]
            v = v / np.linalg.norm(v)
            if not np.isfinite(v[2]):
                plunge = 0.0
                azimuth = 0.0
            else:
                plunge = float(np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0))))
                azimuth = float((np.rad2deg(np.arctan2(v[0], v[1])) + 360) % 360)

            confidence_ellipsoid = ConfidenceEllipsoid(
                semi_major_axis_length=s1 * 1000.0,
                semi_minor_axis_length=s3 * 1000.0,
                semi_intermediate_axis_length=s2 * 1000.0,
                major_axis_plunge=plunge,
                major_axis_azimuth=azimuth,
            )

            origin_uncertainty = OriginUncertainty(
                confidence_ellipsoid=confidence_ellipsoid,
                horizontal_uncertainty=ERH_1s * scale95 * 1000.0,
                preferred_description="confidence ellipsoid",
                confidence_level=95.0,
            )
            origin.origin_uncertainty = origin_uncertainty

            depth_errors = QuantityError(
                uncertainty=ERZ_1s * scale95 * 1000.0,
                confidence_level=95.0,
            )
            origin.depth_errors = depth_errors
            origin.depth_type = (
                "operator assigned"
                if data.fixed_depth_mask[event_idx]
                else "from location"
            )

            # Populate arrival residuals
            for station_idx, (network, station) in enumerate(data.stations):
                for phase_idx, phase in enumerate(config.phases):
                    arr_index = config.n_phases * station_idx + phase_idx
                    observed_time_rel = data.arrivals[event_idx, arr_index]
                    if np.isnan(observed_time_rel):
                        continue

                    # Find the corresponding pick in the event
                    pick = next(
                        (
                            pk
                            for pk in event.picks
                            if pk.waveform_id.station_code == station
                            and pk.waveform_id.network_code == network
                            and (pk.phase_hint or "").lower() == phase.lower()
                        ),
                        None,
                    )

                    # Theoretical travel time (seconds)
                    travel_time = data.theoretical_travel_times[event_idx, arr_index]

                    # Observed travel time = observed arrival (REL) - t0_rel
                    observed_travel_time = (
                        observed_time_rel - data.origin_times[event_idx]
                    )

                    # Residual with the same sign convention as inversion:
                    # residual = observed - theoretical
                    residual = observed_travel_time - travel_time

                    arrival = Arrival(
                        pick_id=pick.resource_id if pick is not None else None,
                        phase=phase,
                        time_residual=float(residual),
                    )
                    origin.arrivals.append(arrival)

            event.origins.append(origin)

        return data.events

    @staticmethod
    def report_invalid_events(data: DataHandler) -> np.ndarray:
        """Return indices of events that were marked invalid during processing."""
        return np.where(data.invalid_events)[0]
