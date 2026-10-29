<div align="center">

<img src="docs/assets/hypy_logo.svg" alt="hypy logo" width="140" height="140" />

# hypy

**Locate seismic events from phase arrivals.**

</div>

## Overview

`hypy` is a Python package for **absolute seismic event location** (hypocenter
determination) from P/S phase picks. It performs iterative least-squares
inversion against a velocity model and reports relocated origins together with
uncertainty estimates.

### Features

- **Extensible forward models** — layered and volumetric models (with precomputed travel-time tables if needed).
- **Multiple phases** — locate using `P`, `S`, or both.
- **Flexible inversion** — damped least-squares (LSQR) inversion, optional depth fixing, and distance-based observation weighting.
- **Uncertainty estimates** — diagonal or full (correlated) data covariance and error-ellipsoid computation.
- **ObsPy-native I/O** — input provided and output served are ObsPy objects.

## Installation

Requires Python $\geq$ 3.11.

```bash
git clone https://github.com/Antcating/hypy
cd hypy
pip install .
```

Optional backends require extra packages, available as
[uv](https://docs.astral.sh/uv/) dependency groups from a source checkout:

```bash
uv sync --group pyrocko  # Pyrocko ray tracer
uv sync --group fmm      # scikit-fmm 1D travel-time tables
uv sync --group 3d       # 3D volumetric models (netcdf4, nllgrid, pyekfmm)
```

With plain `pip`, install the corresponding packages instead
(`pyrocko`, `scikit-fmm`, `netcdf4`, `nllgrid`, `pyekfmm`).

## Usage

```python
from obspy import read_inventory, read_events

from hypy import EventLocator
from hypy.model.layered.tvel import TvelModel
from hypy.propagation.layered.ray_tracer.hypo import HypoRayTracer

# 1. Load a 1D velocity model (e.g. a built-in TauP model exported as .tvel)
model = TvelModel.from_file("iasp91.tvel")

# 2. Initiate a propagation engine (here, a port of HypoInverse ray tracer))
engine = HypoRayTracer(model)

# 3. Configure the locator
locator = EventLocator(engine, phases=("P", "S"))

# 4. Load station metadata and event picks (ObsPy objects)
inventory = read_inventory("stations.xml")
catalog = read_events("picks.xml")
locator.load_data(inventory, catalog)

# 5. Locate — returns relocated origins and indices of events that failed
located_catalog, invalid_indices = locator.locate()
```

Each successfully located event gets an ObsPy `Origin` (with latitude,
longitude, depth, origin time, and uncertainties) appended to it.

### Key `EventLocator` options

| Argument | Description | Default |
| --- | --- | --- |
| `phases` | Phases to use (`"P"`, `"S"`). | `("p", "s")` |
| `derivative_method` | `"numerical"` or `"analytical"`. | `"numerical"` |
| `distance_weight_start_km` / `distance_weight_end_km` | Distance-based observation weighting window (a `locate()` option). | `None` |
| `min_picks_per_event` | Minimum picks required to attempt a location. | `4` |

`locate()` accepts `max_iterations`, `conv_threshold`,
`trial_depth_km`, `unresolved_depth`, distance/residual weighting
options, `model_uncertainty_s`, and more; `fix_depth_km` is an
`EventLocator` constructor option — see the documentation for details.

## Project structure

```
hypy/
├── locator/        # EventLocator orchestration: data, inversion, errors, export
├── model/          # Velocity models (layered .tvel, volumetric netCDF/NLLGrid)
├── propagation/    # Propagation engines (layered ray tracers, grid/table backends)
└── utils/          # Geodesy and math helpers
```

## Testing

```bash
uv sync --group dev   # or: pip install pytest pytest-cov
pytest
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for
enhancements or bug fixes.

## Known issues

Equidistant geometries have an ambiguity trade-off between depth and origin
time, so the inversion may not converge to the correct source parameters
(depth and origin time). This can be mitigated by providing a good initial
guess of the source parameters.

## License

`hypy` is released under the MIT License.
