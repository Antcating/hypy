from typing import List, Tuple
import numpy as np
from obspy.taup import TauPyModel
from obspy.geodetics import kilometer2degrees, gps2dist_azimuth
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.core.event import Pick, Event
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
import os


class EventLocator:
    def __init__(
        self,
        model: TauPyModel,
        dim: int = 3,
        phases: List[str] = ["P"],
        max_depth: float = 20.0,
        max_distance: float = 100.0,
        depth_step: float = 0.2,
        distance_step: float = 0.5,
    ):
        """
        Initialize the EventLocator with a velocity model and configure travel-time tables.

        Parameters:
        - model: TauPyModel representing the velocity structure (e.g., IASP91 model).
        - dim: Dimensionality of the location problem (3 for latitude, longitude, depth).
               (For 2D location, use dim=2; depth will be fixed or not solved).
        - phases: List of seismic phase labels to use (e.g., ["P"] or ["P", "S"]).
        - max_depth, max_distance: Maximum depth (km) and epicentral distance (km) covered by travel-time tables.
        - depth_step, distance_step: Grid spacing for depth and distance in the precomputed tables.
        """
        self.model = model
        self.dim = dim
        self.phases = phases
        self.n_phases = len(phases)

        # Precompute travel-time and ray-parameter tables for specified phases.
        self.depth_grid = None
        self.distance_grid = None
        self.travel_time_tables = {}  # Travel-time tables for each phase
        self.ray_param_tables = {}  # Ray-parameter (slowness) tables for each phase
        self.takeoff_angle_tables = {}
        self._build_travel_time_tables(
            max_depth, max_distance, depth_step, distance_step
        )

        # Placeholders for inventory and events data (to be loaded with load_data()).
        self.inv: Inventory = None
        self.networks: List[str] = []
        self.stations: List[Tuple[str, str]] = []
        self.stations_locations: List[Tuple[float, float, float]] = []
        self.n_stations: int = 0
        self.events: List[Event] = []
        self.n_events: int = 0

        # Arrays for origin times, observed arrivals, theoretical arrivals, derivative matrix, and locations.
        self.origin_times: np.ndarray = None
        self.arrivals: np.ndarray = None
        self.theoretical_arrivals: np.ndarray = None
        self.derivative_matrix: lil_matrix = None
        self.locations: np.ndarray = None

    def _travel_time_table_filename(
        self, max_depth, max_distance, depth_step, distance_step
    ):
        # Generate a filename for cached travel-time tables based on phases and table dimensions.
        phase_str = "_".join(sorted(self.phases))
        return f"travel_time_tables_{phase_str}_d{max_depth}_r{max_distance}_dz{depth_step}_dr{distance_step}.npz"

    def save_travel_time_tables(self, filename: str):
        """Save the travel-time and ray-parameter tables to a compressed NumPy file."""
        if self.depth_grid is None or self.distance_grid is None:
            raise ValueError("Travel-time tables are not built; nothing to save.")
        # Prepare dictionary of tables to save (prefix tt_ for travel times, rp_ for ray params).
        save_dict = {"depth_grid": self.depth_grid, "distance_grid": self.distance_grid}
        for ph in self.travel_time_tables:
            save_dict[f"tt_{ph}"] = self.travel_time_tables[ph]
        for ph in self.ray_param_tables:
            save_dict[f"rp_{ph}"] = self.ray_param_tables[ph]
        for ph in self.takeoff_angle_tables:
            save_dict[f"ta_{ph}"] = self.takeoff_angle_tables[ph]
        np.savez_compressed(filename, **save_dict)

    def load_travel_time_tables(self, filename: str):
        """Load travel-time and ray-parameter tables from a file created by save_travel_time_tables()."""
        data = np.load(filename, allow_pickle=True)
        self.depth_grid = data["depth_grid"]
        self.distance_grid = data["distance_grid"]
        self.travel_time_tables = {}
        self.ray_param_tables = {}
        self.takeoff_angle_tables = {}
        for ph in self.phases:
            if f"tt_{ph}" in data:
                self.travel_time_tables[ph] = data[f"tt_{ph}"]
            if f"rp_{ph}" in data:
                self.ray_param_tables[ph] = data[f"rp_{ph}"]
            if f"ta_{ph}" in data:
                self.takeoff_angle_tables[ph] = data[f"ta_{ph}"]

    def _build_travel_time_tables(
        self,
        max_depth: float,
        max_distance: float,
        depth_step: float,
        distance_step: float,
        overwrite: bool = False,
    ):
        """
        Build (or load, if available) the travel-time and ray-parameter tables for the given phases.
        The tables cover depths from 0 to max_depth (km) and distances from 0 to max_distance (km).
        """
        fname = self._travel_time_table_filename(
            max_depth, max_distance, depth_step, distance_step
        )
        if os.path.exists(fname) and not overwrite:
            # Load precomputed tables from file to save time.
            print(f"Loading travel-time tables from {fname}")
            self.load_travel_time_tables(fname)
            return
        # Initialize depth and distance grids.
        depth_values = np.arange(0.0, max_depth + 1e-6, depth_step)
        distance_values = np.arange(0.0, max_distance + 1e-6, distance_step)
        self.depth_grid = depth_values
        self.distance_grid = distance_values
        # Initialize empty tables (filled with NaN) for each phase.
        for phase in self.phases:
            self.travel_time_tables[phase] = np.full(
                (len(depth_values), len(distance_values)), np.nan
            )
            self.ray_param_tables[phase] = np.full(
                (len(depth_values), len(distance_values)), np.nan
            )
            self.takeoff_angle_tables[phase] = np.full(
                (len(depth_values), len(distance_values)), np.nan
            )
        # Fill the tables by querying the TauPy model for each depth-distance point.
        for i, depth in enumerate(depth_values):
            for j, dist in enumerate(distance_values):
                # Convert distance in km to degrees for TauPyModel queries.
                dist_deg = kilometer2degrees(dist)
                # For each phase, get the first arrival travel time and ray parameter at this depth & distance.
                # We use phase codes "ttp" and "tts" for direct P and S (to get the first P or S arrival).
                if "P" in self.phases or "p" in self.phases:
                    if "P" in self.travel_time_tables:
                        try:
                            arrivals = self.model.get_travel_times(
                                source_depth_in_km=depth,
                                distance_in_degree=dist_deg,
                                phase_list=["ttp"],
                            )
                        except Exception as e:
                            arrivals = []
                        if arrivals:
                            arr = arrivals[0]
                            if hasattr(arr, "time"):
                                self.travel_time_tables["P"][i, j] = arr.time
                            if hasattr(arr, "ray_param"):
                                self.ray_param_tables["P"][i, j] = arr.ray_param
                            if hasattr(arr, "takeoff_angle"):
                                self.takeoff_angle_tables["P"][i, j] = arr.takeoff_angle
                if "S" in self.phases or "s" in self.phases:
                    if "S" in self.travel_time_tables:
                        try:
                            arrivals = self.model.get_travel_times(
                                source_depth_in_km=depth,
                                distance_in_degree=dist_deg,
                                phase_list=["tts"],
                            )
                        except Exception as e:
                            arrivals = []
                        if arrivals:
                            arr = arrivals[0]
                            if hasattr(arr, "time"):
                                self.travel_time_tables["S"][i, j] = arr.time
                            if hasattr(arr, "ray_param"):
                                self.ray_param_tables["S"][i, j] = arr.ray_param
                            if hasattr(arr, "takeoff_angle"):
                                self.takeoff_angle_tables["S"][i, j] = arr.takeoff_angle
        # Basic sanity check to ensure tables are populated (not all NaNs).
        if "P" in self.phases:
            assert not np.all(np.isnan(self.travel_time_tables["P"])), (
                "Travel-time table for 'P' is all NaN."
            )
            assert not np.all(np.isnan(self.ray_param_tables["P"])), (
                "Ray-parameter table for 'P' is all NaN."
            )
        if "S" in self.phases:
            assert not np.all(np.isnan(self.travel_time_tables["S"])), (
                "Travel-time table for 'S' is all NaN."
            )
            assert not np.all(np.isnan(self.ray_param_tables["S"])), (
                "Ray-parameter table for 'S' is all NaN."
            )
        # Optionally save the tables to disk for reuse.
        print(f"Saving travel-time tables to {fname}")
        self.save_travel_time_tables(fname)

    def load_data(self, inv: Inventory, events: List[Event]):
        """
        Load station inventory (station metadata) and events (with picks) into the locator.
        This must be called before running locate().

        Parameters:
        - inv: ObsPy Inventory containing Network/Station information (stations used for picking).
        - events: List of ObsPy Event objects, each with Pick entries for station arrival times.
        """
        self.inv = inv
        # Flatten station list from inventory (collect all station codes and their locations).
        self.networks = [net.code for net in inv]
        self.stations = [(net.code, sta.code) for net in inv for sta in net.stations]
        self.stations_locations = [
            (sta.latitude, sta.longitude, sta.elevation)
            for net in inv
            for sta in net.stations
        ]
        self.n_stations = len(self.stations)
        # Store events and count them.
        self.events = events
        self.n_events = len(events)
        # Allocate data arrays for these events and stations.
        self._allocate_arrays(self.n_events, self.n_stations, self.n_phases)
        # Fill the observed arrival times matrix from the event picks.
        self._fill_data()

    def _allocate_arrays(self, n_events: int, n_stations: int, n_phases: int):
        """Allocate arrays for origin times, observed arrivals, theoretical arrivals, derivative matrix, and locations."""
        # Origin times (to be adjusted during iteration)
        self.origin_times = np.zeros(n_events)
        # Observed arrival times (picks) for each event [shape: n_events x (n_phases * n_stations)], initialize with NaN.
        self.arrivals = np.full((n_events, n_phases * n_stations), np.nan)
        # Theoretical arrival times (predicted from current location estimate)
        self.theoretical_arrivals = np.zeros((n_events, n_phases * n_stations))
        # Partial derivative (Jacobian) matrix of shape (n_events * n_phases * n_stations) x (n_events * dim).
        self.derivative_matrix = lil_matrix(
            (n_events * n_phases * n_stations, n_events * self.dim)
        )
        # Estimated event locations (lat, lon, depth for each event). Initialized to zeros.
        self.locations = np.zeros((n_events, self.dim))

    def _fill_data(self):
        """Populate the arrivals matrix with observed pick times from each event."""
        for event_idx, event in enumerate(self.events):
            for station_idx, (network, station) in enumerate(self.stations):
                for phase_idx, phase in enumerate(self.phases):
                    # Find a Pick in this event corresponding to the station and phase.
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
                    arrival_time = np.nan if pick is None else pick.time
                    # Fill the observed arrival time (UTCDateTime) into the array (will be used in residual calculation).
                    self.arrivals[
                        event_idx, self.n_phases * station_idx + phase_idx
                    ] = arrival_time

    def _interpolate_takeoff_angle(
        self, phase: str, depth: float, distance: float
    ) -> float:
        """
        Interpolate the takeoff angle (degrees) from the precomputed table.
        """
        if self.depth_grid is None or self.distance_grid is None:
            raise ValueError("Takeoff angle table is not initialized.")

        # Clamp depth/distance
        depth = max(min(depth, self.depth_grid[-1]), self.depth_grid[0])
        distance = max(min(distance, self.distance_grid[-1]), self.distance_grid[0])
        i1 = np.searchsorted(self.depth_grid, depth)
        i0 = i1 - 1
        if i0 < 0:
            i0 = 0
        if i1 >= len(self.depth_grid):
            i1 = len(self.depth_grid) - 1
        if i0 == i1:
            return self._interp_takeoff_angle_one_depth(phase, i0, distance)
        t0 = self._interp_takeoff_angle_one_depth(phase, i0, distance)
        t1 = self._interp_takeoff_angle_one_depth(phase, i1, distance)
        d0, d1 = self.depth_grid[i0], self.depth_grid[i1]
        if d1 == d0:
            return t0
        return t0 + (depth - d0) / (d1 - d0) * (t1 - t0)

    def _interpolate_travel_time(
        self, phase: str, depth: float, distance: float
    ) -> float:
        """
        Interpolate the travel time (seconds) for a given phase, source depth (km), and distance (km) from the precomputed table.
        Uses linear interpolation in depth and parabolic interpolation in distance for accuracy:contentReference[oaicite:3]{index=3}.
        """
        if self.depth_grid is None or self.distance_grid is None:
            raise ValueError("Travel-time tables not initialized.")
        # Clamp depth and distance within table bounds.
        depth = max(min(depth, self.depth_grid[-1]), self.depth_grid[0])
        distance = max(min(distance, self.distance_grid[-1]), self.distance_grid[0])
        # Find indices around the given depth.
        i1 = np.searchsorted(self.depth_grid, depth)
        i0 = i1 - 1
        if i0 < 0:
            i0 = 0
        if i1 >= len(self.depth_grid):
            i1 = len(self.depth_grid) - 1
        # If depth lies exactly on a grid point, interpolate only in distance domain at that depth index.
        if i0 == i1:
            return self._interp_time_one_depth(phase, i0, distance)
        # Otherwise, interpolate travel time at the two bracketing depths, then interpolate between them.
        t0 = self._interp_time_one_depth(phase, i0, distance)
        t1 = self._interp_time_one_depth(phase, i1, distance)
        # Linear interpolation in depth.
        depth0, depth1 = self.depth_grid[i0], self.depth_grid[i1]
        if depth1 == depth0:
            return t0
        frac = (depth - depth0) / (depth1 - depth0)
        return t0 + frac * (t1 - t0)

    def _interp_takeoff_angle_one_depth(
        self, phase: str, depth_index: int, distance: float
    ) -> float:
        angle_row = self.takeoff_angle_tables[phase][depth_index]
        dist_grid = self.distance_grid
        if distance <= dist_grid[0]:
            return angle_row[0]
        if distance >= dist_grid[-1]:
            j0 = len(dist_grid) - 2
            j1 = len(dist_grid) - 1
            x0, x1 = dist_grid[j0], dist_grid[j1]
            y0, y1 = angle_row[j0], angle_row[j1]
            return y1 + (distance - x1) * (y1 - y0) / (x1 - x0)
        j1 = np.searchsorted(dist_grid, distance)
        j0 = j1 - 1
        if j0 < 0:
            j0 = 0
        if j1 >= len(dist_grid):
            j1 = len(dist_grid) - 1
        if dist_grid[j0] == distance or j0 == j1:
            return angle_row[j0]
        if j0 >= 1 and j1 < len(dist_grid) - 1:
            if (distance - dist_grid[j0]) < (dist_grid[j1] - distance):
                idxs = [j0 - 1, j0, j1]
            else:
                idxs = [j0, j1, j1 + 1]
        else:
            idxs = [j0, j1]
        if len(idxs) == 2:
            x0, x1 = dist_grid[idxs[0]], dist_grid[idxs[1]]
            y0, y1 = angle_row[idxs[0]], angle_row[idxs[1]]
            return y0 + (distance - x0) / (x1 - x0) * (y1 - y0)
        jA, jB, jC = idxs
        xA, xB, xC = dist_grid[jA], dist_grid[jB], dist_grid[jC]
        yA, yB, yC = angle_row[jA], angle_row[jB], angle_row[jC]
        denomA = (xA - xB) * (xA - xC)
        denomB = (xB - xA) * (xB - xC)
        denomC = (xC - xA) * (xC - xB)
        return (
            yA * (distance - xB) * (distance - xC) / denomA
            + yB * (distance - xA) * (distance - xC) / denomB
            + yC * (distance - xA) * (distance - xB) / denomC
        )

    def _interp_time_one_depth(
        self, phase: str, depth_index: int, distance: float
    ) -> float:
        """
        Interpolate travel time at a specific depth index for a given distance (km).
        Uses linear interpolation or three-point parabolic interpolation in the distance dimension:contentReference[oaicite:4]{index=4}.
        """
        times_row = self.travel_time_tables[phase][
            depth_index
        ]  # travel times for this depth at all distances
        dist_grid = self.distance_grid
        # If distance is out of table range, do linear extrapolation using the nearest two points.
        if distance <= dist_grid[0]:
            return times_row[0]
        if distance >= dist_grid[-1]:
            j0 = len(dist_grid) - 2
            j1 = len(dist_grid) - 1
            x0, x1 = dist_grid[j0], dist_grid[j1]
            y0, y1 = times_row[j0], times_row[j1]
            if x1 == x0:
                return y1
            # Linear extrapolation beyond the farthest distance.
            return y1 + (distance - x1) * (y1 - y0) / (x1 - x0)
        # Find nearest indices around the distance.
        j1 = np.searchsorted(dist_grid, distance)
        j0 = j1 - 1
        if j0 < 0:
            j0 = 0
            j1 = 0
        if j1 >= len(dist_grid):
            j1 = len(dist_grid) - 1
            j0 = j1
        # If distance exactly matches a table point, return that directly.
        if dist_grid[j0] == distance or j0 == j1:
            return times_row[j0]
        # Determine interpolation range for distance:
        if j0 >= 1 and j1 < len(dist_grid) - 1:
            # If possible, use a three-point parabolic interpolation (for smooth derivative).
            # Choose three points such that the target distance lies between the first and last.
            if (distance - dist_grid[j0]) < (dist_grid[j1] - distance):
                idxs = [j0 - 1, j0, j1]
            else:
                idxs = [j0, j1, j1 + 1]
        else:
            # Near the edges, fall back to linear interpolation.
            idxs = [j0, j1]
        if len(idxs) == 2:
            # Linear interpolation between idxs[0] and idxs[1].
            x0, x1 = dist_grid[idxs[0]], dist_grid[idxs[1]]
            y0, y1 = times_row[idxs[0]], times_row[idxs[1]]
            if x1 == x0:
                return y0
            return y0 + (distance - x0) / (x1 - x0) * (y1 - y0)
        # Parabolic interpolation using three points (Lagrange polynomial).
        jA, jB, jC = idxs
        xA, xB, xC = dist_grid[jA], dist_grid[jB], dist_grid[jC]
        yA, yB, yC = times_row[jA], times_row[jB], times_row[jC]
        denomA = (xA - xB) * (xA - xC)
        denomB = (xB - xA) * (xB - xC)
        denomC = (xC - xA) * (xC - xB)
        # Lagrange polynomial terms:
        termA = yA * (distance - xB) * (distance - xC) / denomA
        termB = yB * (distance - xA) * (distance - xC) / denomB
        termC = yC * (distance - xA) * (distance - xB) / denomC
        return termA + termB + termC

    def _interpolate_ray_param(
        self, phase: str, depth: float, distance: float
    ) -> float:
        """
        Interpolate the ray parameter (s/radian) for a given phase, source depth, and distance from the precomputed table.
        Uses the same method as travel time interpolation, ensuring consistency with travel-time derivatives.
        """
        if self.depth_grid is None or self.distance_grid is None:
            raise ValueError("Travel-time tables not initialized.")
        # Clamp inputs to table range.
        depth = max(min(depth, self.depth_grid[-1]), self.depth_grid[0])
        distance = max(min(distance, self.distance_grid[-1]), self.distance_grid[0])
        # Find indices around the given depth.
        i1 = np.searchsorted(self.depth_grid, depth)
        i0 = i1 - 1
        if i0 < 0:
            i0 = 0
        if i1 >= len(self.depth_grid):
            i1 = len(self.depth_grid) - 1
        if i0 == i1:
            return self._interp_ray_param_one_depth(phase, i0, distance)
        # Interpolate ray parameter at bracketing depths.
        p0 = self._interp_ray_param_one_depth(phase, i0, distance)
        p1 = self._interp_ray_param_one_depth(phase, i1, distance)
        depth0, depth1 = self.depth_grid[i0], self.depth_grid[i1]
        if depth1 == depth0:
            return p0
        # Linear interpolation in depth for ray parameter.
        frac = (depth - depth0) / (depth1 - depth0)
        return p0 + frac * (p1 - p0)

    def _interp_ray_param_one_depth(
        self, phase: str, depth_index: int, distance: float
    ) -> float:
        """
        Interpolate ray parameter at a specific depth index for the given distance (km).
        Uses linear/parabolic interpolation in the distance dimension (analogous to travel time interpolation).
        """
        params_row = self.ray_param_tables[phase][depth_index]
        dist_grid = self.distance_grid
        # Handle out-of-bound distances by linear extrapolation.
        if distance <= dist_grid[0]:
            return params_row[0]
        if distance >= dist_grid[-1]:
            j0 = len(dist_grid) - 2
            j1 = len(dist_grid) - 1
            x0, x1 = dist_grid[j0], dist_grid[j1]
            y0, y1 = params_row[j0], params_row[j1]
            if x1 == x0:
                return y1
            return y1 + (distance - x1) * (y1 - y0) / (x1 - x0)
        # Find nearest indices around the distance.
        j1 = np.searchsorted(dist_grid, distance)
        j0 = j1 - 1
        if j0 < 0:
            j0 = 0
            j1 = 0
        if j1 >= len(dist_grid):
            j1 = len(dist_grid) - 1
            j0 = j1
        if dist_grid[j0] == distance or j0 == j1:
            return params_row[j0]
        # Determine if we can use 3-point interpolation in distance.
        if j0 >= 1 and j1 < len(dist_grid) - 1:
            if (distance - dist_grid[j0]) < (dist_grid[j1] - distance):
                idxs = [j0 - 1, j0, j1]
            else:
                idxs = [j0, j1, j1 + 1]
        else:
            idxs = [j0, j1]
        if len(idxs) == 2:
            # Linear interpolation for ray parameter.
            x0, x1 = dist_grid[idxs[0]], dist_grid[idxs[1]]
            y0, y1 = params_row[idxs[0]], params_row[idxs[1]]
            if x1 == x0:
                return y0
            return y0 + (distance - x0) / (x1 - x0) * (y1 - y0)
        # Parabolic interpolation for ray parameter (using three neighboring points).
        jA, jB, jC = idxs
        xA, xB, xC = dist_grid[jA], dist_grid[jB], dist_grid[jC]
        yA, yB, yC = params_row[jA], params_row[jB], params_row[jC]
        denomA = (xA - xB) * (xA - xC)
        denomB = (xB - xA) * (xB - xC)
        denomC = (xC - xA) * (xC - xB)
        termA = yA * (distance - xB) * (distance - xC) / denomA
        termB = yB * (distance - xA) * (distance - xC) / denomB
        termC = yC * (distance - xA) * (distance - xB) / denomC
        return termA + termB + termC

    def _derive_travel_time_param(self, event_location, station_location, phase):
        ev_lat, ev_lon, ev_depth = event_location
        st_lat, st_lon, st_elev = station_location

        # Distance and azimuth
        dist_m, az_deg, _ = gps2dist_azimuth(ev_lat, ev_lon, st_lat, st_lon)
        distance_km = dist_m / 1000.0

        # Travel time (interpolated)
        travel_time = self._interpolate_travel_time(
            phase, ev_depth + st_elev / 1000, distance_km
        )

        # Takeoff angle (degrees, interpolated from table)
        takeoff_angle = self._interpolate_takeoff_angle(
            phase, ev_depth + st_elev / 1000, distance_km
        )

        # Velocity at source depth
        vel = self.model.model.s_mod.v_mod.evaluate_below(ev_depth, phase.lower())[0]

        # Derivatives in radial and vertical direction
        takeoff_rad = np.deg2rad(takeoff_angle)
        dt_dr = np.sin(takeoff_rad) / vel
        dt_dz = -np.cos(takeoff_rad) / vel  # depth positive downward

        # Project into lat/lon using azimuth
        az_rad = np.deg2rad(az_deg)
        dt_dx = -dt_dr * np.cos(az_rad)
        dt_dy = -dt_dr * np.sin(az_rad)

        return travel_time, dt_dx, dt_dy, dt_dz

    def init_locations(self, trial_depth: float = 10.0, trial_time: float = 2.0):
        """
        Initialize the event locations and origin times for iterative location.
        For each event, use the station of the earliest arriving pick as the initial guess for epicenter,
        with a given trial depth. Set the origin time so that the first pick's travel time is roughly trial_time.
        """
        for event_idx, event in enumerate(self.events):
            if len(event.picks) == 0:
                # If no picks for this event, skip initialization.
                continue
            # Find the earliest pick (minimum time) for this event.
            first_pick = min(event.picks, key=lambda pk: pk.time)
            net_code = first_pick.waveform_id.network_code
            sta_code = first_pick.waveform_id.station_code
            # Get station coordinates from the inventory.
            station = self.inv.select(network=net_code, station=sta_code)[0][0]
            # Initialize event location at that station's coordinates, with trial_depth.
            self.locations[event_idx] = (
                station.latitude,
                station.longitude,
                trial_depth,
            )
            # Initialize origin time so that travel time to that station is ~ trial_time.
            self.origin_times[event_idx] = first_pick.time - trial_time
        return self.locations

    def locate(
        self,
        max_iterations: int = 10,
        conv_threshold: float = 1e-4,
        solve_airquakes: bool = False,
        max_depth: float = 20.0,
        max_depth_iterations: int = 5,
        damping_list: List[float] = None,
    ):
        """
        Perform iterative earthquake location for all loaded events using a linearized least-squares approach.

        Parameters:
        - max_iterations: Maximum number of iteration steps.
        - conv_threshold: Convergence threshold in km for the largest location update.
        - solve_airquakes: If False, prevent solutions with negative depth (above surface). If True, allow negative depths.
        - max_depth: Maximum depth allowed. Events that try to go deeper than this (after max_depth_iterations) are flagged invalid.
        - max_depth_iterations: Number of iterations after which the max_depth constraint is enforced.
        - damping_list: Optional list of damping values (Tikhonov regularization) to apply for each iteration (length can equal iterations).

        Returns:
        A tuple (locations, origin_times, max_update) after convergence or reaching max_iterations.
        """
        # Initialize event locations and origin times.
        self.init_locations()
        prev_max_update = np.inf
        if damping_list is None:
            damping_list = [0.0] * max_iterations  # no damping by default
        # Track invalid events (e.g., events forced out of physical bounds).
        self.invalid_events = np.zeros(self.n_events, dtype=bool)

        for it in range(max_iterations):
            valid_rows = []  # list to collect row indices in the matrix that have valid data
            # Loop over events and stations to compute predicted arrivals and partial derivatives.
            for event_idx in range(self.n_events):
                if self.invalid_events[event_idx]:
                    continue  # skip events already marked invalid
                ev_lat, ev_lon, ev_depth = self.locations[event_idx]
                for station_idx in range(self.n_stations):
                    sta_lat, sta_lon, sta_elev = self.stations_locations[station_idx]
                    for phase_idx, phase in enumerate(self.phases):
                        col_idx = (
                            event_idx * self.dim
                        )  # start index of this event's parameters in solution vector
                        arr_index = (
                            self.n_phases * station_idx + phase_idx
                        )  # index of this pick in the arrivals array
                        observed_time = self.arrivals[event_idx, arr_index]
                        if np.isnan(observed_time):
                            continue  # no pick for this event-station-phase combination
                        # Compute predicted travel time and derivatives for this event-station-phase.
                        travel_time, dt_dx, dt_dy, dt_dz = (
                            self._derive_travel_time_param(
                                (ev_lat, ev_lon, ev_depth),
                                (sta_lat, sta_lon, sta_elev),
                                phase,
                            )
                        )
                        self.theoretical_arrivals[event_idx, arr_index] = travel_time
                        # Fill the derivative (Jacobian) matrix.
                        row_idx = (
                            event_idx * (self.n_stations * self.n_phases) + arr_index
                        )
                        if self.dim >= 2:
                            self.derivative_matrix[row_idx, col_idx + 0] = (
                                dt_dx  # ∂t/∂lat
                            )
                            self.derivative_matrix[row_idx, col_idx + 1] = (
                                dt_dy  # ∂t/∂lon
                            )
                        if self.dim == 3:
                            self.derivative_matrix[row_idx, col_idx + 2] = (
                                dt_dz  # ∂t/∂depth
                            )
                        valid_rows.append(row_idx)

            # Compute residuals: observed travel times (pick time - origin_time) vs. theoretical travel times.
            travel_times_obs = (
                self.arrivals - self.origin_times[:, np.newaxis]
            )  # observed travel time for each pick
            residuals = (
                self.theoretical_arrivals - travel_times_obs
            )  # residual = predicted - observed
            residuals_flat = residuals.flatten()
            valid_rows = np.array(valid_rows, dtype=int)
            # Assemble the vector of residuals and corresponding rows of the derivative matrix.
            b = residuals_flat[valid_rows]  # residual vector for valid picks
            A = self.derivative_matrix.tocsr()[
                valid_rows, :
            ]  # sparse Jacobian matrix for valid picks

            # Remove mean residual per event (decoupling origin time from spatial coordinates).
            # This is analogous to how HypoInverse adjusts origin time each iteration:contentReference[oaicite:10]{index=10}.
            if self.n_events > 0:
                residual_mean = np.zeros(self.n_events)
                for event_idx in range(self.n_events):
                    # Identify indices in b corresponding to this event.
                    event_rows = [
                        idx
                        for idx, row in enumerate(valid_rows)
                        if (row // (self.n_stations * self.n_phases)) == event_idx
                    ]
                    if event_rows:
                        mean_res = np.mean(b[event_rows])
                        residual_mean[event_idx] = mean_res
                        # Subtract mean residual from those rows (so that average residual per event becomes zero).
                        b[event_rows] -= mean_res
                # Adjust origin times by adding the mean residual (improving the fit of origin time).
                self.origin_times -= residual_mean

            # Solve the linear system A * dx = b for location adjustments dx (in least-squares sense, possibly damped).
            damping = damping_list[it] if it < len(damping_list) else damping_list[-1]
            dx_solution = lsqr(A, b, damp=damping, atol=1e-6, btol=1e-6, iter_lim=1000)[
                0
            ]
            dx = dx_solution.reshape((self.n_events, self.dim))

            # Limit the magnitude of updates to avoid divergence (especially important for early iterations).
            max_step_xy = 3.0  # max horizontal step in km per iteration
            max_step_z = 2.0  # max depth step in km per iteration
            for d in range(min(self.dim, 3)):
                if d < 2:
                    # Clip latitude/longitude updates (in km).
                    dx[:, d] = np.clip(dx[:, d], -max_step_xy, max_step_xy)
                else:
                    # Clip depth updates.
                    dx[:, d] = np.clip(dx[:, d], -max_step_z, max_step_z)

            # Check for any events going out of bounds (negative depth or too deep) and mark them invalid if so.
            for event_idx in range(self.n_events):
                if self.invalid_events[event_idx]:
                    continue
                new_depth = self.locations[event_idx, 2] - (
                    dx[event_idx, 2] if self.dim == 3 else 0.0
                )
                # If not allowing negative depths and solution suggests a negative depth, or if beyond max_depth after some iterations:
                if (not solve_airquakes and new_depth < 0.0) or (
                    it > max_depth_iterations and new_depth > max_depth
                ):
                    # Mark event as invalid (unable to converge within physical depth bounds).
                    print(
                        f"Event {event_idx} would have depth {new_depth:.2f} km out of bounds. Marking as invalid."
                    )
                    self.invalid_events[event_idx] = True
                    # Nullify this iteration's update for that event.
                    if self.dim == 3:
                        dx[event_idx, 2] = 0.0

            # Convert horizontal (lat, lon) updates from km back to degrees for updating location coordinates.
            if self.dim >= 2:
                dx[:, 0] = kilometer2degrees(dx[:, 0])  # lat: km -> degrees
                dx[:, 1] = kilometer2degrees(dx[:, 1])  # lon: km -> degrees
            # (Depth updates remain in km, as depth is handled in km units directly.)

            # Compute the maximum update magnitude (in km for horizontal and depth) among all events this iteration.
            max_update = 0.0
            if dx.size > 0:
                valid_mask = (
                    ~self.invalid_events
                )  # only consider valid events for convergence
                if np.any(valid_mask):
                    max_update = np.max(np.abs(dx[valid_mask]))

            # Update event locations with the computed adjustments.
            self.locations -= dx
            # (Origin times were already adjusted above via mean residual removal.)

            # Check convergence criteria:
            if max_update < conv_threshold:
                print(
                    f"Converged after {it + 1} iterations (max update {max_update:.3f} km < {conv_threshold} km)."
                )
                break
            # If the improvement is very small (change in max_update < 0.001% between iterations) and max_update is modest, consider converged (to avoid oscillation).
            if (
                prev_max_update != 0
                and abs(prev_max_update - max_update) / prev_max_update < 1e-5
                and max_update < 0.5
            ):
                print(
                    f"Converged (minor change) after {it + 1} iterations (max update {max_update:.3f} km)."
                )
                break
            prev_max_update = max_update

        return self.locations, self.origin_times, max_update

    def report_invalid_events(self) -> np.ndarray:
        """
        Return an array of indices for events that were marked as invalid (e.g., requiring non-physical depths).
        """
        if hasattr(self, "invalid_events"):
            return np.where(self.invalid_events)[0]
        return np.array([])


if __name__ == "__main__":
    from obspy.taup import TauPyModel
    from obspy.core import UTCDateTime
    from obspy.core.event import Event, Pick, WaveformStreamID

    # Load a TauPy model (e.g., IASP91).
    model = TauPyModel(model="iasp91")

    # Define synthetic stations and their locations
    station_coords = {
        "STA01": (10.0, 10.0, 0.0),
        "STA02": (10.05, 10.0, 0.0),
        "STA03": (10.25, 10.0, 0.0),
        "STA04": (10.0, 10.05, 0.0),
        "STA05": (10.0, 10.25, 0.0),
        "STA07": (10.07, 10.07, 0.0),
        "STA08": (10.15, 10.15, 0.0),
        "STA09": (10.2, 10.2, 0.0),
        "STA10": (10.3, 10.3, 0.0),
    }
    # Build inventory
    net = Network(code="NET", start_date=UTCDateTime("2020-01-01"))
    for sta, (lat, lon, elev) in station_coords.items():
        net.stations.append(
            Station(
                code=sta,
                latitude=lat,
                longitude=lon,
                elevation=elev,
                start_date=UTCDateTime("2020-01-01"),
                site=Site(name=sta),
            )
        )
    inv = Inventory(networks=[net], source="synthetic")
    # Define synthetic event
    event_latlon_depth = (10.05, 10.05, 10.0)  # lat, lon, depth in km
    origin_time = UTCDateTime("2024-01-01T01:01:01.100Z")
    # Generate synthetic picks
    event = Event()
    for sta, (lat, lon, elev) in station_coords.items():
        arrivals = model.get_travel_times_geo(
            source_latitude_in_deg=event_latlon_depth[0],
            source_longitude_in_deg=event_latlon_depth[1],
            source_depth_in_km=event_latlon_depth[2],
            receiver_latitude_in_deg=lat,
            receiver_longitude_in_deg=lon,
            phase_list=["tts"],
        )
        if not arrivals:
            print(f"No arrivals for station {sta}, skipping.")
            continue
        arr = arrivals[0]
        pick_time = origin_time + arr.time
        pick = Pick(
            time=pick_time,
            phase_hint="S",
            waveform_id=WaveformStreamID(
                network_code="NET",
                station_code=sta,
                channel_code="ENZ",
            ),
        )
        event.picks.append(pick)
    events = [event]

    locator = EventLocator(model, dim=3, phases=["S"])
    locator.load_data(inv, events)
    locator.init_locations()
    locations, origin_times, _ = locator.locate(
        max_iterations=500,
    )

    print("Final event locations:")
    for i, loc in enumerate(locations):
        lat, lon, depth = loc
        print(
            f"Event {i}: Latitude: {lat:.4f}, Longitude: {lon:.4f}, Depth: {depth:.2f} km"
        )
        print(f"Origin time: {origin_times[i]}")
