Getting started
=============== 

``hypy`` is a Python package for absolute seismic event location (hypocenter determination) 
from P/S phase picks. It performs iterative least-squares inversion against a velocity 
model and reports relocated origins together with uncertainty estimates.


The public API is **composition based**: you assemble three objects and wire them together:

1. A **velocity model** (a :class:`~hypy.model.VelocityModel` subclass) that
   describes the Earth structure.
2. A **propagation engine** (a :class:`~hypy.propagation.base.PropagationEngine`
   subclass) that computes travel times and their spatial derivatives for a
   given model.
3. An :class:`~hypy.locator.EventLocator` that runs the inversion using that
   engine.

   model  ──►  engine  ──►  EventLocator  ──►  located catalog

Minimal example
----------------

.. code-block:: python

   import obspy

   from hypy.locator import EventLocator
   from hypy.model.layered.tvel import TvelModel
   from hypy.propagation.layered.ray_tracer.hypo import HypoRayTracer

   # 1. Load a 1D velocity model from a .tvel file.
   model = TvelModel("model/giimod.tvel")

   # 2. Build a propagation engine on top of that model.
   engine = HypoRayTracer(model)

   # 3. Configure the locator with the engine instance.
   locator = EventLocator(engine, phases=("P", "S"))

   # 4. Load station metadata and event picks (ObsPy objects).
   inventory = obspy.read_inventory("input/stations.xml")
   catalog = obspy.read_events("input/picks.xml")
   locator.load_data(inventory, catalog)

   # 5. Locate. Returns the relocated catalog and indices of failed events.
   located_catalog, invalid_indices = locator.locate(solver="lsqr")

   print(located_catalog[0].origins[-1])

Each successfully located event receives an ObsPy ``Origin`` (latitude,
longitude, depth, origin time, and uncertainty estimates) appended to it.

Choosing a propagation engine
------------------------------

Different engines trade accuracy for speed. They all consume a
:class:`~hypy.model.VelocityModel` and expose the same runtime contract, so
they are interchangeable from the locator's point of view.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Engine
     - Import
     - Notes
   * - ``HypoRayTracer``
     - ``hypy.propagation.layered.ray_tracer.hypo``
     - Fast local 1D ray tracing (port of the HypoInverse tracer).
   * - ``TauPRayTracer``
     - ``hypy.propagation.layered.ray_tracer.taup``
     - 1D ray tracing through ObsPy TauP.
   * - ``PyrockoRayTracer``
     - ``hypy.propagation.layered.ray_tracer.pyrocko``
     - 1D ray tracing through Pyrocko (requires ``[pyrocko]``).
   * - ``TauTravelTimeGrid``
     - ``hypy.propagation.layered.grid.tau``
     - Precomputed 1D travel-time tables (fast repeated lookups).
   * - ``FmmTravelTimeGrid1D``
     - ``hypy.propagation.layered.grid.fmm``
     - 1D fast-marching travel-time grid (requires ``[fmm]``).
   * - ``FmmTravelTimeGrid3D``
     - ``hypy.propagation.volumetric.grid.pyekfmm``
     - 3D fast-marching travel-time grid (requires ``[3d]``).

Table-based engines need grid extents at construction time, for example::

   from hypy.propagation.layered.grid.tau import TauTravelTimeGrid

   engine = TauTravelTimeGrid(
       model,
       phases=["P", "S"],
       max_depth_km=100.0,
       max_distance_km=300.0,
       depth_step_km=1.0,
       distance_step_km=1.0,
       engine="taup",
   )

A 3D volumetric model is wired the same way::

   from hypy.model.volumetric.netcdf import NCGrid
   from hypy.propagation.volumetric.grid.pyekfmm import FmmTravelTimeGrid3D

   model = NCGrid("model/3d_Vp_Vs_VpVs_models.nc")
   engine = FmmTravelTimeGrid3D(
       model,
       phases=["P", "S"],
       max_depth_km=100.0,
       max_distance_km=300.0,
       depth_step_km=1.0,
       distance_step_km=1.0,
   )

Key ``EventLocator`` options
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Argument
     - Description
     - Default
   * - ``phases``
     - Phases to use (``"P"``, ``"S"``).
     - ``("P", "S")``
   * - ``initial_propagation_engine``
     - Optional coarse/table engine for seed scoring; if omitted, heuristic/random seeds are used.
     - ``None``
   * - ``derivative_method``
     - ``"numerical"`` or ``"analytical"``. Must be supported by the engine.
     - ``"numerical"``
   * - ``fix_depth_km``
     - Fix event depth during inversion.
     - ``None``
   * - ``min_picks_per_event``
     - Minimum picks required to attempt a location.
     - ``4``

``locate()`` accepts ``solver`` (``"lsqr"`` / ``"svd"``), ``max_iterations``,
``conv_threshold``, ``trial_depth_km``, ``solve_airquakes`` (if false, an
airquake means depth is unresolved and is fixed at the seed hypocenter), and
``safe_iteration_limit``. It also takes the per-run knobs that were previously
constructor arguments, so they can be varied between runs on the same loaded
data: ``distance_weight_start_km`` / ``distance_weight_end_km`` (distance-based
observation weighting window), ``full_data_covariance`` /
``correlation_length_km`` (correlated observation covariance for uncertainty
estimation), ``error_coefficient``, and ``seed``.

Station time corrections
------------------------

If a station is known to be late (clock delay, site static), pass a delay
table so the inversion subtracts that delay from the observed arrival.
Positive values mean the observed arrival is late. Load the table however
you like (typically ``pandas.read_csv``) and pass the DataFrame:

.. code-block:: python

   import pandas as pd

   delays = pd.read_csv("input/station_delays.csv")
   #    network station phase  delay_s
   # 0      IU    ANMO   NaN     0.15
   # 1      IU    ANMO     S     0.25

   locator = EventLocator(engine, phases=("P", "S"))
   locator.load_data(inventory, catalog, station_time_corrections=delays)

Required columns are ``network``, ``station``, and ``delay_s``. Optional
``phase`` (missing / NaN / ``*``) applies to all configured phases; a
phase row overrides the station-wide value. Stations listed in the table
but missing from the loaded inventory/picks are ignored. A Python mapping
``{("IU", "ANMO"): 0.15, ("IU", "ANMO", "S"): 0.25}`` is also accepted.

Compatibility checks
--------------------

When the engine is built and again when it is passed to ``EventLocator``,
``hypy`` validates that the pieces fit together:

* The engine's :attr:`required_model_features` must be a subset of the model's
  :attr:`~hypy.model.VelocityModel.feature_flags`; otherwise the engine
  constructor raises ``ValueError``.
* The engine's :attr:`feature_flags` must advertise the requested derivative
  capability (``derivative.numerical`` or ``derivative.analytical``);
  otherwise ``EventLocator`` raises ``ValueError``.

These checks fail fast, before any inversion runs
