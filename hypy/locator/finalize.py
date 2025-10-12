import numpy as np

from .types import (
    TYPE_CHECKING,
    Arrival,
    Catalog,
    ConfidenceEllipsoid,
    Event,
    List,
    Origin,
    OriginUncertainty,
    QuantityError,
    Union,
)  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from .state import LocatorState


class FinalizeMixin:
    def _finalize_origins(self: "LocatorState") -> Union[List[Event], Catalog]:
        scale95 = 2.4
        for event_idx, event in enumerate(self.events):
            # If invalid - no origins are added
            if self.invalid_events[event_idx]:
                continue
            origin = Origin()
            origin.latitude = self.locations[event_idx, 0]
            origin.longitude = self.locations[event_idx, 1]
            origin.depth = self.locations[event_idx, 2] * 1000.0
            origin.time = self.origin_times[event_idx]
            s1, s2, s3 = self._principal_std[event_idx]
            evecs = self._principal_axes[event_idx]
            ERZ_1s, ERH_1s = self._erz_erh[event_idx]

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
                if self.fixed_depth_mask[event_idx]
                else "from location"
            )

            for station_idx, (network, station) in enumerate(self.stations):
                for phase_idx, phase in enumerate(self.phases):
                    arr_index = self.n_phases * station_idx + phase_idx
                    observed_time = self.arrivals[event_idx, arr_index]
                    if np.isnan(observed_time):
                        continue
                    pick = next(
                        (
                            pk
                            for pk in event.picks
                            if pk.waveform_id.station_code == station
                            and pk.waveform_id.network_code == network
                            and pk.phase_hint.lower() == phase.lower()
                        ),
                        None,
                    )
                    travel_time = self.theoretical_travel_times[event_idx, arr_index]
                    residual = travel_time - (
                        observed_time - self.origin_times[event_idx]
                    )
                    arrival = Arrival(
                        pick_id=pick.resource_id, phase=phase, time_residual=residual
                    )
                    origin.arrivals.append(arrival)
            event.origins.append(origin)
        return self.events

    def report_invalid_events(self: "LocatorState") -> np.ndarray:
        if hasattr(self, "invalid_events"):
            return np.where(self.invalid_events)[0]
        return np.array([])
