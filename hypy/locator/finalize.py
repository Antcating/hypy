import numpy as np

from .types import (
    Arrival,
    Catalog,
    ConfidenceEllipsoid,
    Event,
    FinalizeContext,
    FinalizeUncertainty,
    List,
    Matrices,
    Origin,
    OriginUncertainty,
    QuantityError,
    Union,
)  # type: ignore[import-untyped]


def finalize_origins(
    *,
    events: Union[List[Event], Catalog],
    matrices: Matrices,
    ctx: FinalizeContext,
    unc: FinalizeUncertainty,
) -> Union[List[Event], Catalog]:
    """Attach final Origin objects (with uncertainties) to each event."""
    scale95 = 2.4
    events_list = list(events)
    for event_idx, event in enumerate(events_list):
        if matrices.invalid_events[event_idx]:
            continue

        origin = Origin()
        origin.latitude = matrices.locations[event_idx, 0]
        origin.longitude = matrices.locations[event_idx, 1]
        origin.depth = matrices.locations[event_idx, 2] * 1000.0  # km -> m

        t0_abs = float(matrices.origin_times[event_idx]) + float(
            matrices.time_refs[event_idx]
        )
        origin.time = t0_abs

        s1, s2, s3 = unc.principal_std[event_idx]
        evecs = unc.principal_axes[event_idx]
        ERZ_1s, ERH_1s = unc.erz_erh[event_idx]

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
            "operator assigned" if ctx.fixed_depth_mask[event_idx] else "from location"
        )

        for station_idx, (network, station) in enumerate(ctx.stations):
            for phase_idx, phase in enumerate(ctx.phases):
                arr_index = ctx.n_phases * station_idx + phase_idx
                observed_time_rel = matrices.arrivals[event_idx, arr_index]
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

                travel_time = matrices.theoretical_travel_times[event_idx, arr_index]
                observed_travel_time = (
                    observed_time_rel - matrices.origin_times[event_idx]
                )
                residual = observed_travel_time - travel_time

                arrival = Arrival(
                    pick_id=pick.resource_id if pick is not None else None,
                    phase=phase,
                    time_residual=float(residual),
                )
                origin.arrivals.append(arrival)

        event.origins.append(origin)

    return events_list


def report_invalid_events(*, invalid_events: np.ndarray) -> np.ndarray:
    return np.where(invalid_events)[0]
