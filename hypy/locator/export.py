"""Export location results to ObsPy catalogs and events."""

import numpy as np
from obspy.core.event import Arrival, Catalog, Origin
from obspy.core.event.base import QuantityError
from obspy.core.event.origin import ConfidenceEllipsoid, OriginUncertainty

# pyrefly: ignore [untyped-import]
from scipy.stats import chi2, norm

from .config import LocatorConfig
from .data import DataHandler
from .types import ErrorResults


def finalize_origins(
    config: LocatorConfig,
    data: DataHandler,
    error_results: ErrorResults,
) -> Catalog:
    """Attach final ObsPy origins and return the resulting catalog."""
    # Confidence-region scaling depends on dimensionality.
    scale95_3d = float(np.sqrt(chi2.ppf(0.95, df=3)))
    scale95_2d = float(np.sqrt(chi2.ppf(0.95, df=2)))
    scale95_1d = float(norm.ppf(0.975))

    for event_idx, event in enumerate(data.events):
        if data.invalid_events[event_idx]:
            continue

        origin = Origin()
        origin.latitude = data.locations[event_idx, 0]
        origin.longitude = data.locations[event_idx, 1]
        origin.depth = data.locations[event_idx, 2] * 1000.0

        t0_abs = float(data.origin_times[event_idx]) + float(data.time_refs[event_idx])
        origin.time = t0_abs

        s1, s2, s3 = error_results.principal_std[event_idx]
        evecs = error_results.principal_axes[event_idx]
        ERZ_1s, ERH_1s = error_results.erz_erh[event_idx]

        # ErrorResults stores one principal-axis vector per ROW (major,
        # intermediate, minor) in (north, east, down) coordinates.
        # QuakeML Tait-Bryan R = Rz(azimuth) Ry(plunge) Rx(rotation) has
        # columns (major, minor, intermediate) with the major axis pointing up.
        R = np.asarray(evecs[[0, 2, 1], :], dtype=float).T.copy()
        col_norms = np.linalg.norm(R, axis=0)
        if not np.all(np.isfinite(R)) or np.any(col_norms <= 0.0):
            plunge = 0.0
            azimuth = 0.0
            rotation = 0.0
        else:
            R /= col_norms
            if R[2, 0] > 0.0:
                R[:, 0] *= -1.0
                R[:, 1] *= -1.0
            if np.linalg.det(R) < 0.0:
                R[:, 1] *= -1.0
            plunge = float(np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))))
            if np.hypot(R[0, 0], R[1, 0]) > 1e-12:
                azimuth = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360.0)
                rotation = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
            else:
                azimuth = 0.0
                rotation = float(np.degrees(np.arctan2(R[0, 1], R[1, 1])))

        # Events whose covariance could not be estimated keep the NaN
        # placeholders from ErrorResults. ObsPy rejects non-finite
        # uncertainties, so report "unknown" by omitting the field.
        confidence_ellipsoid = None
        if np.all(np.isfinite((s1, s2, s3))):
            confidence_ellipsoid = ConfidenceEllipsoid(
                semi_major_axis_length=s1 * scale95_3d * 1000.0,
                semi_minor_axis_length=s3 * scale95_3d * 1000.0,
                semi_intermediate_axis_length=s2 * scale95_3d * 1000.0,
                major_axis_plunge=plunge,
                major_axis_azimuth=azimuth,
                major_axis_rotation=rotation,
            )

        origin.origin_uncertainty = OriginUncertainty(
            confidence_ellipsoid=confidence_ellipsoid,
            horizontal_uncertainty=(
                ERH_1s * scale95_2d * 1000.0 if np.isfinite(ERH_1s) else None
            ),
            preferred_description="confidence ellipsoid",
            confidence_level=95.0,
        )

        if np.isfinite(ERZ_1s):
            origin.depth_errors = QuantityError(
                uncertainty=ERZ_1s * scale95_1d * 1000.0,
                confidence_level=95.0,
            )
        origin.depth_type = (
            "operator assigned" if data.fixed_depth_mask[event_idx] else "from location"
        )

        pick_lookup = {
            (
                pk.waveform_id.network_code,
                pk.waveform_id.station_code,
                (pk.phase_hint or "").lower(),
            ): pk
            for pk in event.picks
        }

        for station_idx, (network, station) in enumerate(data.stations):
            for phase_idx, phase in enumerate(config.phases):
                arr_index = config.n_phases * station_idx + phase_idx
                observed_time_rel = data.arrivals[event_idx, arr_index]
                if np.isnan(observed_time_rel):
                    continue

                pick = pick_lookup.get((network, station, phase.lower()))

                travel_time = data.theoretical_travel_times[event_idx, arr_index]
                observed_travel_time = observed_time_rel - data.origin_times[event_idx]
                time_correction = float(data.station_time_corrections[arr_index])
                residual = observed_travel_time - travel_time - time_correction

                # Final weight includes distance and residual scales.
                # time_error_W is kept separate (ObsPy Arrival has no sigma field
                # here); residual_W is the last-iteration HypoInverse-style factor.
                row_index = (
                    event_idx * config.n_phases * data.n_stations
                    + config.n_phases * station_idx
                    + phase_idx
                )
                final_weight = data.distance_W[row_index] * data.residual_W[row_index]

                arrival = Arrival(
                    pick_id=pick.resource_id if pick is not None else None,
                    phase=phase,
                    time_residual=float(residual),
                    time_weight=float(final_weight),
                    time_correction=time_correction,
                )
                origin.arrivals.append(arrival)

        event.origins.append(origin)

    return Catalog(events=list(data.events))
