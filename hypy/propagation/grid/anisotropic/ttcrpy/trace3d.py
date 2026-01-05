"""
travel_time_tables.py
--------------------
Handles travel-time, ray-parameter, and takeoff angle table creation, saving, loading, and interpolation.
"""

from typing import List, Tuple

import numpy as np
from obspy.taup import TauPyModel  # type: ignore[import-untyped]
import ttcrpy.rgrid as rg

from hypy.propagation import PropagationModel
from hypy.utils.geo import gps2dist_azimuth_nb


class Trace3D(PropagationModel):
    """
    Manages travel-time, ray-parameter, and takeoff angle tables for seismic phases.
    """

    def __init__(
        self,
        model_tvel: str,
        model_name: str,
        phases: List[str],
        max_depth: float,
        max_distance: float,
        depth_step: float,
        distance_step: float,
        model_center: Tuple[float, float],
    ):
        self.model = TauPyModel(model_name)
        self.model_name = model_name
        self.phases = [phase.lower() for phase in phases]
        self.max_depth = max_depth
        self.max_distance = max_distance
        self.depth_step = depth_step
        self.distance_step = distance_step
        self.model_center = model_center

        assert (
            self.distance_step == self.depth_step
        ), "Distance step must equal depth step for 3D grid."

        self.x_grid = np.arange(
            0,
            self.max_distance + self.distance_step,
            self.distance_step,
        )
        self.y_grid = np.arange(
            0,
            self.max_distance + self.distance_step,
            self.distance_step,
        )

        self.z_grid = np.arange(0, self.max_depth + self.depth_step, self.depth_step)
        self.V = {
            phase.lower(): np.empty(
                (self.x_grid.size, self.y_grid.size, self.z_grid.size)
            )
            for phase in self.phases
        }
        self.grid = rg.Grid3d(
            self.x_grid,
            self.y_grid,
            self.z_grid,
            cell_slowness=False,
            method="FSM",
            n_threads=4,
            eps=1e-3,
            maxit=3,
        )

        self._build_grid()

    def _build_grid(self) -> None:
        """
        Build (or load) the travel-time, ray-parameter, and takeoff angle tables for the given phases.
        """
        for phase in self.phases:
            phase = phase.lower()
            # Temporary generator for 1D velocity
            for depth_idx, depth in enumerate(
                np.arange(0, self.max_depth + self.depth_step, self.depth_step)
            ):
                v = self.model.model.s_mod.v_mod.evaluate_below(depth, phase)
                self.V[phase][:, :, depth_idx] = v
        self.slowness = {phase: 1.0 / self.V[phase] for phase in self.phases}

    def dT_dsource_from_trace(self, path_xyz, v_source, nseg=3):
        """
        path_xyz: (N,3) points from SOURCE -> RECEIVER.
        v_source: km/s at the source (grid value).
        Returns (dT/dx_s, dT/dy_s, dT/dz_s) in s/km.
        """
        if path_xyz.shape[0] < 2:
            raise ValueError("Ray path too short")

        nseg = max(1, min(nseg, path_xyz.shape[0] - 1))

        d = np.zeros(3, dtype=np.float64)
        for i in range(nseg):
            seg = path_xyz[i + 1] - path_xyz[i]
            n = np.linalg.norm(seg)
            if n > 0:
                d += seg / n

        nd = np.linalg.norm(d)
        if nd == 0:
            raise ValueError("Degenerate ray direction near source")

        u_sr = d / nd  # unit direction: source -> receiver
        p = u_sr / v_source  # slowness vector at source (s/km)
        return (-p[0], -p[1], -p[2])  # derivative wrt SOURCE location

    def compute_travel_parameters(
        self,
        source_lon: float,
        source_lat: float,
        source_depth_km: float,
        receiver_lon: float,
        receiver_lat: float,
        receiver_elev_m: float,
        phase: str,
    ) -> "tuple[float, float, tuple[float, float, float]]":
        """
        Get arrival parameters for all phases between source and receiver.
        """
        dist_m, azimuth_deg, _ = gps2dist_azimuth_nb(
            source_lat,
            source_lon,
            receiver_lat,
            receiver_lon,
            R=self.model.model.radius_of_planet * 1000.0,
        )
        distance_km = dist_m / 1000.0

        source_depth_km += receiver_elev_m / 1000.0  # account for station elevation

        src_center_dist, src_center_az, _ = gps2dist_azimuth_nb(
            self.model_center[0], self.model_center[1], source_lat, source_lon
        )
        src_center_dist_km = src_center_dist / 1000

        src = np.array(
            [
                [
                    src_center_dist_km * np.cos(np.deg2rad(src_center_az))
                    + self.max_distance / 2,
                    src_center_dist_km * np.sin(np.deg2rad(src_center_az))
                    + self.max_distance / 2,
                    source_depth_km,
                ]
            ]
        )

        rcv_center_dist, rcv_center_az, _ = gps2dist_azimuth_nb(
            self.model_center[0], self.model_center[1], receiver_lat, receiver_lon
        )
        rcv_center_dist_km = rcv_center_dist / 1000

        rcv = np.array(
            [
                [
                    rcv_center_dist_km * np.cos(np.deg2rad(rcv_center_az))
                    + self.max_distance / 2,
                    rcv_center_dist_km * np.sin(np.deg2rad(rcv_center_az))
                    + self.max_distance / 2,
                    0.0,
                ]
            ]
        )

        tt, rays = self.grid.raytrace(
            src, rcv, self.slowness[phase.lower()], return_rays=True
        )
        travel_time = tt[0]
        ray = rays[0][::-1]
        dt_dx, dt_dy, dt_dz = self.dT_dsource_from_trace(
            ray, self.V[phase.lower()][0, 0, int(source_depth_km / self.depth_step)]
        )

        return distance_km, travel_time, (dt_dx, dt_dy, dt_dz)
