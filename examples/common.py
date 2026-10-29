import math
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io.img_tiles import OSM
from matplotlib.patches import Ellipse
from obspy.core.event import ConfidenceEllipsoid, Event, OriginUncertainty
from obspy.core.inventory import Inventory


def horizontal_projection_ellipse(
    semi_major_m,
    semi_minor_m,
    semi_intermediate_m,
    major_axis_plunge_deg,
    major_axis_azimuth_deg,
    major_axis_rotation_deg,
):
    a = semi_major_m
    b = semi_minor_m
    c = semi_intermediate_m

    az = math.radians(major_axis_azimuth_deg)
    pl = math.radians(major_axis_plunge_deg)
    rot = math.radians(major_axis_rotation_deg)

    ca, sa = math.cos(az), math.sin(az)
    cp, sp = math.cos(pl), math.sin(pl)
    cr, sr = math.cos(rot), math.sin(rot)

    # QuakeML Tait-Bryan R = Rz(azimuth) Ry(plunge) Rx(rotation) in NED,
    # columns = (major, minor, intermediate); permute to ENU.
    R_ned = np.array(
        [
            [ca * cp, ca * sp * sr - sa * cr, ca * sp * cr + sa * sr],
            [sa * cp, sa * sp * sr + ca * cr, sa * sp * cr - ca * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )
    R = np.array([R_ned[1], R_ned[0], -R_ned[2]], dtype=float)

    # 3D ellipsoid shape matrix
    S3 = R @ np.diag([a * a, b * b, c * c]) @ R.T

    # Orthogonal projection onto horizontal x-y plane
    S2 = S3[:2, :2]

    # Extract horizontal ellipse axes
    eigvals, eigvecs = np.linalg.eigh(S2)
    order = np.argsort(eigvals)[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    horizontal_semi_major_m = math.sqrt(eigvals[0])
    horizontal_semi_minor_m = math.sqrt(eigvals[1])

    # Horizontal azimuth of projected semi-major axis
    v = eigvecs[:, 0]  # [east, north]
    azimuth_deg = math.degrees(math.atan2(v[0], v[1])) % 180.0

    return {
        "horizontal_semi_major_m": horizontal_semi_major_m,
        "horizontal_semi_minor_m": horizontal_semi_minor_m,
        "horizontal_major_axis_azimuth_deg": azimuth_deg,
        "horizontal_shape_matrix": S2,
    }


def plot_solution(
    inventory: Inventory,
    event: Event,
    save_path: Optional[str] = None,
    use_cartopy: bool = False,
    tile_zoom: int = 10,
    **kwargs,
):
    origin = event.origins[-1]
    dpi = kwargs.get("dpi", 150)
    station_coords = [
        (station.latitude, station.longitude)
        for network in inventory.networks
        for station in network.stations
    ]
    st_lats = [lat for lat, lon in station_coords]
    st_lons = [lon for lat, lon in station_coords]

    if use_cartopy:
        projection = ccrs.PlateCarree()
        fig, ax = plt.subplots(
            figsize=(6, 6),
            dpi=dpi,
            subplot_kw={"projection": projection},
        )
        tiler = OSM()
        ax.add_image(tiler, tile_zoom)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="none")

        ax.scatter(
            st_lons,
            st_lats,
            marker="o",
            s=30,
            color="blue",
            label="Stations",
            transform=projection,
            zorder=4,
        )
        ax.scatter(
            [origin.longitude],
            [origin.latitude],
            facecolors="k",
            marker="x",
            s=60,
            label="Estimated origin",
            transform=projection,
            zorder=6,
        )

        margin = 0.1
        ax.set_extent(
            [
                min([origin.longitude]) - margin,
                max([origin.longitude]) + margin,
                min([origin.latitude]) - margin,
                max([origin.latitude]) + margin,
            ],
            crs=projection,
        )
        ax.gridlines(draw_labels=True, linestyle=":")
    else:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.scatter(st_lons, st_lats, marker="o", s=10, label="Stations", zorder=2)
        ax.scatter(
            [origin.longitude],
            [origin.latitude],
            facecolors="k",
            marker="x",
            s=40,
            label="Estimated origin",
            zorder=4,
        )

    origin_uncertainty: OriginUncertainty = origin.origin_uncertainty
    confidence_level = origin_uncertainty.confidence_level

    confidence_ellipsoid: ConfidenceEllipsoid = origin_uncertainty.confidence_ellipsoid
    assert confidence_ellipsoid is not None

    horizontal_projection = horizontal_projection_ellipse(
        confidence_ellipsoid.semi_major_axis_length,
        confidence_ellipsoid.semi_minor_axis_length,
        confidence_ellipsoid.semi_intermediate_axis_length,
        confidence_ellipsoid.major_axis_plunge,
        confidence_ellipsoid.major_axis_azimuth,
        confidence_ellipsoid.major_axis_rotation,
    )

    semi_major_km = horizontal_projection["horizontal_semi_major_m"] / 1000.0
    semi_minor_km = horizontal_projection["horizontal_semi_minor_m"] / 1000.0
    azimuth = horizontal_projection["horizontal_major_axis_azimuth_deg"]

    print(
        f"Horizontal confidence ellipse: semi-major={semi_major_km:.4f} km, "
        f"semi-minor={semi_minor_km:.4f} km, azimuth={azimuth:.1f}°"
    )

    if semi_major_km > 0 and semi_minor_km > 0:
        lat0 = origin.latitude
        lat0_rad = np.deg2rad(lat0)

        dlat = semi_minor_km / 111.32
        dlon = semi_major_km / (111.32 * np.cos(lat0_rad))

        width_deg = 2 * dlon
        height_deg = 2 * dlat
        matplotlib_angle = 90.0 - azimuth
        ellipse = Ellipse(
            (origin.longitude, origin.latitude),
            width=width_deg,
            height=height_deg,
            angle=matplotlib_angle,
            facecolor="none",
            edgecolor="red",
            linewidth=2,
            zorder=10,
            label=(
                f"{confidence_level:.0f}% confidence ellipse"
                if confidence_level is not None
                else "Confidence ellipse"
            ),
            transform=projection if use_cartopy else None,
        )
        ax.add_patch(ellipse)

    if not use_cartopy:
        margin = 0.1
        ax.set_xlim(
            min([origin.longitude]) - margin,
            max([origin.longitude]) + margin,
        )
        ax.set_ylim(
            min([origin.latitude]) - margin,
            max([origin.latitude]) + margin,
        )
        ax.grid(True, linestyle=":")
    ax.set_xlabel(r"Longitude (\degree)")
    ax.set_ylabel(r"Latitude (\degree)")
    ax.set_title("Estimated event location")
    ax.legend(loc="best", frameon=True)

    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        print(f"Saved location plot to {save_path}")
    plt.close(fig)


def plot_arrivals_with_weights(
    inventory: Inventory,
    event: Event,
    origin_index: int = -1,
    save_path: Optional[str] = None,
    use_cartopy: bool = False,
    tile_zoom: int = 10,
    **kwargs,
):
    """Plot arrivals colored by their weights (time_weight).

    Parameters
    ----------
    inventory : Inventory
        ObsPy Inventory with station metadata.
    event : Event
        ObsPy Event with origins and arrivals.
    origin_index : int, optional
        Index of origin to plot (default -1 for last origin).
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    use_cartopy : bool, optional
        Use Cartopy for map background (default False).
    tile_zoom : int, optional
        Zoom level for Cartopy tiles (default 10).
    **kwargs
        Additional arguments (e.g., dpi for figure resolution).
    """
    dpi = kwargs.get("dpi", 150)
    origin = event.origins[origin_index]

    # Extract station coordinates
    station_map = {}
    for network in inventory.networks:
        for station in network.stations:
            key = (network.code, station.code)
            station_map[key] = (station.latitude, station.longitude)

    # Extract arrival weights and coordinates
    arrival_lats = []
    arrival_lons = []
    arrival_weights = []
    arrival_labels = []

    for arrival in origin.arrivals:
        if arrival.pick_id is None:
            continue

        # Find the corresponding pick to get station info
        pick = None
        for pk in event.picks:
            if pk.resource_id == arrival.pick_id:
                pick = pk
                break

        if pick is None:
            continue

        network = pick.waveform_id.network_code
        station = pick.waveform_id.station_code
        key = (network, station)

        if key not in station_map:
            continue

        lat, lon = station_map[key]
        arrival_lats.append(lat)
        arrival_lons.append(lon)

        # Get arrival weight (normalized for visualization)
        weight = arrival.time_weight if arrival.time_weight is not None else 1.0
        arrival_weights.append(weight)
        arrival_labels.append(f"{network}.{station}")

    if not arrival_lats:
        print("No arrivals with valid weights found.")
        return

    arrival_weights = np.array(arrival_weights)

    # Get weight range for display
    weight_min = arrival_weights.min()
    weight_max = arrival_weights.max()

    if use_cartopy:
        projection = ccrs.PlateCarree()
        fig, ax = plt.subplots(
            figsize=(8, 8),
            dpi=dpi,
            subplot_kw={"projection": projection},
        )
        tiler = OSM()
        ax.add_image(tiler, tile_zoom)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="none")

        # Plot event origin
        ax.scatter(
            [origin.longitude],
            [origin.latitude],
            facecolors="k",
            marker="*",
            s=200,
            label="Event origin",
            transform=projection,
            zorder=6,
        )

        # Plot arrivals with weights as colors
        scatter = ax.scatter(
            arrival_lons,
            arrival_lats,
            c=arrival_weights,
            s=100,
            cmap="viridis",
            marker="o",
            label="Arrivals",
            transform=projection,
            zorder=4,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.0,
        )

        margin = 0.05
        all_lons = arrival_lons + [origin.longitude]
        all_lats = arrival_lats + [origin.latitude]
        ax.set_extent(
            [
                min(all_lons) - margin,
                max(all_lons) + margin,
                min(all_lats) - margin,
                max(all_lats) + margin,
            ],
            crs=projection,
        )
        ax.gridlines(draw_labels=True, linestyle=":")
    else:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

        # Plot event origin
        ax.scatter(
            [origin.longitude],
            [origin.latitude],
            facecolors="k",
            marker="*",
            s=200,
            label="Event origin",
            zorder=6,
        )

        # Plot arrivals with weights as colors
        scatter = ax.scatter(
            arrival_lons,
            arrival_lats,
            c=arrival_weights,
            s=100,
            cmap="viridis",
            marker="o",
            label="Arrivals",
            zorder=4,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        margin = 0.05
        all_lons = arrival_lons + [origin.longitude]
        all_lats = arrival_lats + [origin.latitude]
        ax.set_xlim(min(all_lons) - margin, max(all_lons) + margin)
        ax.set_ylim(min(all_lats) - margin, max(all_lats) + margin)
        ax.grid(True, linestyle=":")

    ax.set_xlabel(r"Longitude (\degree)")
    ax.set_ylabel(r"Latitude (\degree)")
    title = f"Arrivals colored by weight (min={weight_min:.4f}, max={weight_max:.4f})"
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Arrival weight")

    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        print(f"Saved arrival weights plot to {save_path}")
