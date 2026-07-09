import numpy as np

from hypy.locator.config import LocatorConfig
from hypy.locator.data import DataHandler
from hypy.locator.derivatives import DerivativesHandler
from hypy.propagation.base import PropagationEngine


class DummyTracer(PropagationEngine):
    @property
    def dim(self) -> int:
        return 1

    def get_arrival(self, *args, **kwargs):
        return 1.0, 1.0, 0.0, 0.0

    def compute_travel_time_params(self, *args, **kwargs):
        return 1.0, 1.0, (0.1, 0.2, 0.3, 0.4)


def test_build_derivative_matrix_uses_precomputed_mask_and_returns_valid_rows():
    config = LocatorConfig(
        phases=("P",),
        min_picks_per_event=1,
        default_time_error=0.5,
        mode="table-1d",
        engine="taup",
        derivative_method="analytical",
        max_depth_km=100.0,
        max_distance_km=250.0,
        depth_step_km=1.0,
        distance_step_km=5.0,
        fix_depth_km=None,
        precalculate_velocities=False,
        distance_weighting_enabled=True,
        k=0.1,
        x0=50.0,
    )

    data = DataHandler(
        phases=config.phases,
        min_picks_per_event=config.min_picks_per_event,
        default_time_error=config.default_time_error,
        dim=config.dim,
        fix_depth_km=config.fix_depth_km,
    )
    data.n_events = 1
    data.n_stations = 1
    data._allocate_arrays(1, 1, 1)
    data.locations[0] = np.array([10.0, 20.0, 5.0], dtype=float)
    data.stations_locations = np.array([[10.5, 20.5, 100.0]], dtype=float)
    data.arrivals[0, 0] = 12.3
    data.fixed_depth_mask = [False]

    tracer = DummyTracer(model_path="/tmp/unused")
    handler = DerivativesHandler(config, data, tracer)

    valid_rows = handler.build_derivative_matrix(data, config)

    assert valid_rows.shape == (1,)
    assert valid_rows[0] == 0
    assert data.theoretical_travel_times[0, 0] == 1.0
    assert np.allclose(data.jacobian_matrix[0, :4], [0.3, 0.2, 0.4, 1.0])
    assert data.distance_W[0] != 0.0
