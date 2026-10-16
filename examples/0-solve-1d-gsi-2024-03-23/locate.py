import obspy

from examples.common import plot_arrivals_with_weights, plot_solution
from hypy.locator import EventLocator
from hypy.model.layered.tvel import TvelModel
from hypy.propagation.layered.grid.tau import TauTravelTimeGrid
from hypy.propagation.layered.ray_tracer.hypo import HypoRayTracer
from hypy.propagation.layered.ray_tracer.taup import TauPRayTracer

phases = ["P", "S"]

model_path = "examples/0-solve-1d-gsi-2024-03-23/model/giimod.tvel"

model = TvelModel(model_path)

# Coarse precomputed table: used for seed scoring.
init_grid = TauTravelTimeGrid(
    model,
    phases=phases,
    max_depth_km=100.0,
    max_distance_km=300.0,
    depth_step_km=5.0,
    distance_step_km=20.0,
    tracer=TauPRayTracer(model=model),
)

# Accurate ray tracer: used for the iterative inversion stage.
hypo = HypoRayTracer(model=model, max_distance_km=300.0, max_depth_km=100.0)

locator = EventLocator(
    propagation_engine=hypo,
    initial_propagation_engine=init_grid,
    derivative_method="analytical",
)

inv = obspy.read_inventory(
    "examples/0-solve-1d-gsi-2024-03-23/input/20240323_earthquake_stations.xml"
)
catalog = obspy.read_events(
    "examples/0-solve-1d-gsi-2024-03-23/input/20240323_earthquake_with_gsi_picks.xml"
)

locator.load_data(inv, catalog)
out, invalid = locator.locate(
    max_iterations=1000,
    trial_depth_km=5.0,
    distance_weight_start_km=100.0,
    distance_weight_end_km=200.0,
)

print(out[0].origins[-1])

print(f"Invalid picks: {len(invalid)}")

# extract final origin
e = out[0]
est_lat = e.origins[-1].latitude
est_lon = e.origins[-1].longitude
est_depth_km = e.origins[-1].depth / 1000.0
est_time = e.origins[-1].time.timestamp

print(
    f"Estimated location: lat={est_lat:.4f}, lon={est_lon:.4f}, depth_km={est_depth_km:.2f}, time={est_time}"
)

out.write(
    "examples/0-solve-1d-gsi-2024-03-23/output/20240323_earthquake_located.xml",
    format="QUAKEML",
)

plot_arrivals_with_weights(
    inv,
    e,
    save_path="examples/0-solve-1d-gsi-2024-03-23/output/20240323_earthquake_located_weights.png",
    use_cartopy=True,
)
plot_solution(
    inv,
    e,
    save_path="examples/0-solve-1d-gsi-2024-03-23/output/20240323_earthquake_located.png",
    use_cartopy=True,
)
