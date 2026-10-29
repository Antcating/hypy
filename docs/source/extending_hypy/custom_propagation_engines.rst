Adding a custom propagation engine
====================================

A propagation engine maps a velocity model and a source–receiver
geometry to a travel time and its spatial derivatives. This page is a
practical how-to; the full contract is summarized in
:doc:`addon_propagation` and in the API documentation of
:class:`hypy.propagation.base.PropagationEngine`.

Choose a base class
-------------------

In most cases you only need to write the forward calculation itself;
``hypy`` handles the rest. Pick the class that matches your model:

* **1D layered model** — subclass
  :class:`~hypy.propagation.layered.LayeredPropagationEngine` and
  implement ``get_arrival_1d(distance, depth, phase)``, returning a
  travel time and takeoff angle. Travel times, geometry, and
  derivatives are provided for you.
* **Precomputed 1D travel-time tables** — subclass
  :class:`~hypy.propagation.layered.grid.LayeredGrid1D` and supply the
  tables; lookups are provided for you.
* **3D model** — subclass
  :class:`~hypy.propagation.volumetric.VolumetricPropagationEngine`
  and implement ``get_arrival``; derivatives are computed numerically
  for you.
* **Anything else** — subclass
  :class:`~hypy.propagation.base.PropagationEngine` and implement
  ``compute_travel_time_params`` directly.

The built-in ray tracers use one further intermediate class,
``LayeredRayTracer``, which adds result caching. You only need it if
you are writing a new ray-tracing backend.

Declare the feature contract
----------------------------

Two class attributes declare compatibility, and both are checked
before any inversion runs:

.. code-block:: python

   feature_flags = frozenset({
       "propagation.layered",
       "propagation.ray",
       "derivative.numerical",
       "derivative.analytical",
   })
   required_model_features = frozenset({"model.profile.1d"})

``required_model_features`` must be a subset of the model's
``feature_flags``; :meth:`~hypy.propagation.base.PropagationEngine.validate_model`
raises ``ValueError`` otherwise. The locator additionally requires the
requested derivative capability (``derivative.numerical`` or
``derivative.analytical``) to be present in ``feature_flags``.
Only advertise ``derivative.batch`` if you also implement the
batched methods (``get_arrival_batch``,
``compute_travel_time_params_batch``); most engines do not need it.

Minimal layered engine
----------------------

For a 1D layered model, only the first-arrival calculation is new.
Everything else — including both derivative methods — is inherited:

.. code-block:: python

   from hypy.model import VelocityModel
   from hypy.propagation.layered import LayeredPropagationEngine


   class MyLayeredEngine(LayeredPropagationEngine):
       feature_flags = frozenset({
           "propagation.layered",
           "propagation.ray",
           "derivative.numerical",
           "derivative.analytical",
       })
       required_model_features = frozenset({"model.profile.1d"})

       def __init__(self, model: VelocityModel):
           super().__init__(model)

       def get_arrival_1d(self, arc_distance_km, source_depth_km, phase):
           # Return (travel_time_s, takeoff_angle_deg) for this geometry,
           # e.g. from your own ray tracer or an external library.
           ...
           return travel_time_s, takeoff_angle_deg

Note that analytical derivatives require the model to provide
``get_velocity_1d(depth_km, phase)``, as
:class:`~hypy.model.layered.tvel.TvelModel` does. If your model cannot
supply a local velocity, advertise only ``derivative.numerical``.

Implementing the base contract directly
---------------------------------------

For non-layered geometries, subclass
:class:`~hypy.propagation.base.PropagationEngine` and implement
``compute_travel_time_params``. It must return
``(distance_km, travel_time_s, (dt_dr, dt_dx, dt_dy, dt_dz))``, with the
derivatives in s/km with respect to source–receiver distance and to
the source coordinates (east, north, depth). The horizontal
source-coordinate derivatives carry the opposite sign to the
path-distance derivative: moving the source toward the receiver
shortens the path.

Conventions the locator relies on:

* Return ``travel_time_s=None`` (and ``None`` derivatives) for an
  unusable path — out of range, outside the grid, no arrival. The
  observation is skipped; raising would abort the whole inversion.
* ``super().__init__(model, ...)`` accepts ``max_depth_km``,
  ``max_distance_km``, ``depth_step_km``, and ``distance_step_km``.
  The locator reads these for seeding, depth bounds, and for deciding
  which observations are in range, so set them whenever they apply.
* Phases passed to the engine are normalized (``"p"``/``"s"``).

Testing a new engine
--------------------

At minimum, verify: (i) construction against a compatible model and
``ValueError`` against an incompatible one; (ii) agreement of numerical
and (where offered) analytical derivatives on a few geometries; and
(iii) an end-to-end ``EventLocator`` run. The self-contained example in
``examples/custom_backend_contract.py`` exercises this contract.
