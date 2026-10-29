Starting guide
==============

This guide covers installation and a first, complete location run.
For the underlying method see :doc:`/theory/inversion`; for writing a
custom forward model see
:doc:`/extending_hypy/custom_propagation_engines`.

Installation
------------

``hypy`` requires Python 3.11 or later. The core dependencies
(NumPy, SciPy, ObsPy, Numba, pandas, Matplotlib, Cartopy) are installed
automatically. From a source checkout:

.. code-block:: bash

   git clone <repository-url> hypy
   cd hypy
   pip install .

Optional backends are not part of the core install. With
`uv <https://docs.astral.sh/uv/>`_ they are available as dependency
groups:

.. code-block:: bash

   uv sync --group 3d       # NetCDF/NLL 3D models and the pyekfmm engine
   uv sync --group fmm      # scikit-fmm 1D travel-time tables
   uv sync --group pyrocko  # Pyrocko ray tracer

With plain ``pip``, install the corresponding packages instead
(``netcdf4``, ``nllgrid``, ``pyekfmm``, ``scikit-fmm``, ``pyrocko``).
Verify the installation with:

.. code-block:: bash

   python -c "import hypy; print(hypy.__file__)"

Concepts
--------

A location run is assembled from three objects:

1. a **velocity model** (:class:`~hypy.model.VelocityModel`), which
   describes the Earth structure;
2. a **propagation engine**
   (:class:`~hypy.propagation.base.PropagationEngine`), which computes
   travel times and spatial derivatives for that model; and
3. an :class:`~hypy.locator.EventLocator`, which performs seeding,
   iterative inversion, and uncertainty estimation.

Observations are supplied as standard ObsPy objects: an ``Inventory``
(station coordinates) and a ``Catalog`` whose events carry phase picks.
Each successfully located event receives a new ObsPy ``Origin`` with
latitude, longitude, depth, origin time, and uncertainty estimates.

Worked example: a 1D location
-----------------------------

The example below follows
``examples/0-solve-1d-gsi-2024-03-23/locate.py``. It locates one event
recorded by the Israeli network in a local 1D model
(``giimod.tvel``), using two engines: a coarse precomputed TauP table
scores the initial hypocenters, and an accurate HypoInverse-style ray
tracer drives the inversion.

.. code-block:: python

   import obspy

   from hypy.locator import EventLocator
   from hypy.model.layered.tvel import TvelModel
   from hypy.propagation.layered.grid.tau import TauTravelTimeGrid
   from hypy.propagation.layered.ray_tracer.hypo import HypoRayTracer
   from hypy.propagation.layered.ray_tracer.taup import TauPRayTracer

   phases = ["P", "S"]

   # 1. Load the 1D velocity model.
   model = TvelModel("examples/0-solve-1d-gsi-2024-03-23/model/giimod.tvel")

   # 2. Coarse precomputed table, used only to score seed hypocenters.
   init_grid = TauTravelTimeGrid(
       model,
       phases=phases,
       max_depth_km=100.0,
       max_distance_km=300.0,
       depth_step_km=5.0,
       distance_step_km=20.0,
       tracer=TauPRayTracer(model=model),
   )

   # 3. Accurate ray tracer, used for the iterative inversion.
   hypo = HypoRayTracer(model=model, max_distance_km=300.0, max_depth_km=100.0)

   # 4. Wire the engines into the locator. Analytical derivatives are
   #    available from both engines, so request them explicitly.
   locator = EventLocator(
       propagation_engine=hypo,
       initial_propagation_engine=init_grid,
       derivative_method="analytical",
   )

   # 5. Load station metadata and picks.
   inv = obspy.read_inventory(
       "examples/0-solve-1d-gsi-2024-03-23/input/20240323_earthquake_stations.xml"
   )
   catalog = obspy.read_events(
       "examples/0-solve-1d-gsi-2024-03-23/input/20240323_earthquake_with_gsi_picks.xml"
   )
   locator.load_data(inv, catalog)

   # 6. Locate. Distance weighting tapers observations between
   #    100 km and 200 km epicentral distance.
   out, invalid = locator.locate(
       max_iterations=1000,
       trial_depth_km=5.0,
       distance_weight_start_km=100.0,
       distance_weight_end_km=200.0,
   )

   print(out[0].origins[-1])
   print(f"Invalid events: {len(invalid)}")

   # 7. Export the relocated catalog.
   out.write(
       "examples/0-solve-1d-gsi-2024-03-23/output/20240323_earthquake_located.xml",
       format="QUAKEML",
   )

Running the example
~~~~~~~~~~~~~~~~~~~

The complete script is included in the repository. From the
repository root, run:

.. code-block:: bash

   python examples/0-solve-1d-gsi-2024-03-23/locate.py

The run loads the cached TauP travel-time tables, locates the event,
and writes the relocated catalog and two figures to
``examples/0-solve-1d-gsi-2024-03-23/output/``. Abbreviated log output:

.. code-block:: text

   ... - hypy - INFO - Loaded 1 events and 67 stations with picks for phases ('p', 's').
   ... - hypy - WARNING - 113 invalid time errors found and clamped to 0.1 s.
   ... - hypy - INFO - Locating 1 events with 67 stations using hypo.
   ... - hypy - INFO - Convergence reached at iteration 19: WRMS=0.223309 s
   ... - hypy - INFO - Located 1 events, 0 events invalid.
   ... - hypy - INFO - Final WRMS = 0.223309 s, Final damping = 10.000000
   Estimated location: lat=32.5014, lon=35.1932, depth_km=5.05, time=1711218956.1455448
   Saved location plot to examples/0-solve-1d-gsi-2024-03-23/output/20240323_earthquake_located.png

The two warnings are expected for this dataset: three picked stations
are absent from the inventory (their picks are ignored), and pick
uncertainties missing from the catalog are floored at 0.1 s.

.. figure:: /_static/example_20240323_located.png
   :width: 80%
   :align: center
   :alt: Estimated event location with 95% confidence ellipse

   Location obtained for the 2024-03-23 event (black cross) with its
   95% confidence ellipse (red). The script also produces
   ``20240323_earthquake_located_weights.png``, which shows the final
   per-arrival weights.

Two points are worth noting. First, the two-engine split is optional:
without ``initial_propagation_engine`` the locator falls back to a
multi-seed heuristic scored by the inversion engine itself
(:doc:`/theory/inversion`). The table is used here because scoring many
candidate hypocenters against interpolated travel times is much cheaper
than ray tracing each one. Second, the engines are interchangeable:
replacing ``HypoRayTracer`` with, for example, ``TauPRayTracer`` changes
the forward model but none of the surrounding code.

``locate()`` returns the relocated catalog and the indices of events
that could not be located (for instance, events left with fewer than
``min_picks_per_event`` usable picks). The final origin of a located
event is ``out[0].origins[-1]``; its ``latitude``, ``longitude``,
``depth`` (in metres), and ``time`` attributes give the hypocenter.

Next steps
----------

* :doc:`quick_start` — engine catalogue and per-run options.
* :doc:`/theory/inversion` — seeding, inversion safeguards, stopping
  criteria, and uncertainty estimation.
* :doc:`/extending_hypy/custom_propagation_engines` — adding a new
  forward-model engine.
