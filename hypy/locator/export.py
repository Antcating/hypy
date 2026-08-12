"""Export location results to ObsPy catalogs and events."""

from pathlib import Path
from typing import List, Union

import numpy as np
from obspy.core.event import Arrival, Catalog, Event, Origin
from obspy.core.event.base import QuantityError
from obspy.core.event.origin import ConfidenceEllipsoid, OriginUncertainty

from .config import LocatorConfig
from .data import DataHandler
from .types import ErrorResults


class OriginExporter:
    """Export event location results to ObsPy Event/Catalog structures."""

    def __init__(self, config: LocatorConfig, data: DataHandler) -> None:
        self.config = config
        self.data = data

    def finalize_origins(
        self,
        error_results: ErrorResults,
        *,
        as_catalog: bool = True,
    ) -> Union[Catalog, List[Event]]:
        """Attach final Origin objects to events and return Catalog or List[Event]."""
        events = self._build_event_origins(error_results)
        if as_catalog:
            return Catalog(events=events)
        return events

    def _build_event_origins(self, error_results: ErrorResults) -> List[Event]:
        config = self.config
        data = self.data
        scale95 = 2.4

        for event_idx, event in enumerate(data.events):
            if data.invalid_events[event_idx]:
                continue

            origin = Origin()
            origin.latitude = data.locations[event_idx, 0]
            origin.longitude = data.locations[event_idx, 1]
            origin.depth = data.locations[event_idx, 2] * 1000.0

            t0_abs = float(data.origin_times[event_idx]) + float(
                data.time_refs[event_idx]
            )
            origin.time = t0_abs

            s1, s2, s3 = error_results.principal_std[event_idx]
            evecs = error_results.principal_axes[event_idx]
            ERZ_1s, ERH_1s = error_results.erz_erh[event_idx]

            v = evecs[:, 0]
            v = v / np.linalg.norm(v)
            if not np.isfinite(v[2]):
                plunge = 0.0
                azimuth = 0.0
                rotation = 0.0
            else:
                plunge = float(np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0))))
                azimuth = float((np.rad2deg(np.arctan2(v[0], v[1])) + 360.0) % 360.0)

                w = evecs[:, 1]
                w = w / np.linalg.norm(w)
                ref = np.array([0.0, 0.0, 1.0], dtype=float)
                ref = ref - np.dot(ref, v) * v
                ref_norm = np.linalg.norm(ref)

                if ref_norm < 1e-12:
                    ref = np.array([0.0, 1.0, 0.0], dtype=float)
                    ref = ref - np.dot(ref, v) * v
                    ref_norm = np.linalg.norm(ref)

                if ref_norm < 1e-12 or not np.all(np.isfinite(w)):
                    rotation = 0.0
                else:
                    ref /= ref_norm
                    w_proj = w - np.dot(w, v) * v
                    w_proj_norm = np.linalg.norm(w_proj)

                    if w_proj_norm < 1e-12:
                        rotation = 0.0
                    else:
                        w_proj /= w_proj_norm
                        rotation = float(
                            (
                                np.rad2deg(
                                    np.arctan2(
                                        np.dot(v, np.cross(ref, w_proj)),
                                        np.dot(ref, w_proj),
                                    )
                                )
                                + 360.0
                            )
                            % 180.0
                        )

            confidence_ellipsoid = ConfidenceEllipsoid(
                semi_major_axis_length=s1 * scale95 * 1000.0,
                semi_minor_axis_length=s3 * scale95 * 1000.0,
                semi_intermediate_axis_length=s2 * scale95 * 1000.0,
                major_axis_plunge=plunge,
                major_axis_azimuth=azimuth,
                major_axis_rotation=rotation,
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

            for station_idx, (network, station) in enumerate(data.stations):
                for phase_idx, phase in enumerate(config.phases):
                    arr_index = config.n_phases * station_idx + phase_idx
                    observed_time_rel = data.arrivals[event_idx, arr_index]
                    if np.isnan(observed_time_rel):
                        continue

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

                    travel_time = data.theoretical_travel_times[event_idx, arr_index]
                    observed_travel_time = (
                        observed_time_rel - data.origin_times[event_idx]
                    )
                    residual = observed_travel_time - travel_time

                    # Final weight includes distance and residual scales.
                    # time_error_W is kept separate (ObsPy Arrival has no sigma field
                    # here); residual_W is the last-iteration HypoInverse-style factor.
                    row_index = (
                        event_idx * config.n_phases * data.n_stations
                        + config.n_phases * station_idx
                        + phase_idx
                    )
                    final_weight = (
                        data.distance_W[row_index] * data.residual_W[row_index]
                    )

                    arrival = Arrival(
                        pick_id=pick.resource_id if pick is not None else None,
                        phase=phase,
                        time_residual=float(residual),
                        time_weight=float(final_weight),
                    )
                    origin.arrivals.append(arrival)

            event.origins.append(origin)

        return data.events

    @staticmethod
    def report_invalid_events(data: DataHandler) -> np.ndarray:
        """Return indices of events that were marked invalid during processing."""
        return np.where(data.invalid_events)[0]


def events_to_catalog(events: Union[List[Event], Catalog]) -> Catalog:
    """Normalize output to an ObsPy Catalog."""
    if isinstance(events, Catalog):
        return events
    return Catalog(events=events)


def save_catalog(
    catalog: Catalog,
    path: Union[str, Path],
    format: str = "QUAKEML",
) -> None:
    """Save ObsPy Catalog to disk in specified format (QUAKEML, RES2D, etc.)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    catalog.write(str(p), format=format)


def save_events(
    events: Union[List[Event], Catalog],
    path: Union[str, Path],
    format: str = "QUAKEML",
) -> None:
    """Save list of ObsPy Event or Catalog to disk."""
    save_catalog(events_to_catalog(events), path=path, format=format)
