Adding a custom propagation engine
==================================

A propagation engine turns a velocity model and a source/receiver geometry into
a travel time and its spatial derivatives. This is the single most important
extension point: the locator's inversion is driven entirely by what the engine
returns.

All engines subclass :class:`hypy.propagation.base.PropagationEngine`. For 1D
layered models, :class:`hypy.propagation.layered.LayeredPropagationEngine`
provides ready-made geometry and derivative helpers and is usually the better
starting point.

The contract
------------

Required class attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``feature_flags: frozenset[str]`` — capabilities advertised by the engine.
  Must include at least one derivative flag:

  * ``derivative.numerical`` to support ``derivative_method="numerical"``
  * ``derivative.analytical`` to support ``derivative_method="analytical"``

* ``required_model_features: frozenset[str]`` — model capabilities the engine
  needs. Validated against the model's ``feature_flags`` at construction time.

Required constructor
~~~~~~~~~~~~~~~~~~~~~~

The engine receives an **instantiated** :class:`~hypy.model.VelocityModel` and
must call ``super().__init__(model, ...)``. The base constructor stores the
model, records grid extents, and runs
:meth:`~hypy.propagation.base.PropagationEngine.validate_model`:

.. code-block:: python

   def __init__(self, model: VelocityModel, **kwargs):
       super().__init__(model)

The base ``__init__`` accepts optional keyword arguments that the locator reads
during ``locate()``: ``max_depth_km``, ``max_distance_km``, ``depth_step_km``,
and ``distance_step_km``. Table-based engines should set these; pure ray
tracers can leave them at ``None``.

Required instance members
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``dim`` property returning ``int`` (``1`` for 1D, ``3`` for 3D).
* ``compute_travel_time_params(...)`` — the method the locator calls for every
  event/station/phase on every iteration:

  .. code-block:: python

     def compute_travel_time_params(
         self,
         source_latitude: float,
         source_longitude: float,
         source_depth_km: float,
         receiver_latitude: float,
         receiver_longitude: float,
         receiver_elevation_m: float,
         phase: str,
         delta: float | None = None,
         derivative_method: str = "analytical",
     ) -> tuple[float, float, tuple[float, float, float, float]]:
         ...

  The return value is exactly:

  * ``distance_km`` — great-circle source/receiver distance (``float``).
  * ``travel_time_s`` — predicted travel time (``float``), or ``None`` when the
    path is unusable.
  * derivatives tuple ``(dt_dr, dt_dx, dt_dy, dt_dz)`` — partial derivatives of
    travel time with respect to radial distance and the source coordinates
    (east ``x``, north ``y``, depth ``z``), in s/km.

  For invalid or out-of-range paths, return ``None`` for the travel time (and
  ``None`` values in the derivatives tuple) or NumPy ``nan`` derivatives. The
  locator skips such observations instead of failing the whole inversion.

Recommended members
~~~~~~~~~~~~~~~~~~~~~

These are part of the base contract and should be implemented for consistency,
even though the locator currently funnels everything through
``compute_travel_time_params``:

* ``get_arrival(...)`` — arc distance, travel time, takeoff angle, azimuth.
* ``compute_derivatives(...)`` — spatial slowness derivatives.

How the locator drives the engine
----------------------------------

During ``EventLocator.locate()``:

* At setup, the locator reads ``engine.feature_flags`` and rejects the run if
  the requested ``derivative.<method>`` flag is missing.
* During inversion, ``compute_travel_time_params`` is called with
  ``delta`` set to the configured finite-difference step and
  ``derivative_method`` set to the locator's configuration.
* The horizontal/vertical derivatives ``(dt_dx, dt_dy, dt_dz)`` populate the
  Jacobian; ``distance_km`` feeds optional distance weighting.
* ``engine.max_depth_km`` and ``engine.distance_step_km`` are read for seeding
  and depth bounds, so leave them as plain attributes (the base class sets
  them).

Example: extend the layered base
---------------------------------

For a 1D model, subclass ``LayeredPropagationEngine`` and implement
``get_arrival_1d``. The base class then assembles distances, azimuths, and both
analytical and numerical derivatives for you via its
``compute_travel_time_params`` / ``compute_derivatives`` implementations.

.. code-block:: python

   from hypy.model import VelocityModel
   from hypy.propagation.layered import LayeredPropagationEngine


   class MyLayeredEngine(LayeredPropagationEngine):
       feature_flags = frozenset(
           {
               "propagation.layered",
               "propagation.ray",
               "derivative.numerical",
               "derivative.analytical",
           }
       )
       required_model_features = frozenset({"model.profile.1d"})

       def __init__(self, model: VelocityModel):
           super().__init__(model)

       def get_arrival_1d(
           self,
           arc_distance_km: float,
           source_depth_km: float,
           phase: str,
       ) -> tuple[float, float]:
           # Return (travel_time_s, takeoff_angle_deg) for the given geometry.
           travel_time_s = ...
           takeoff_angle_deg = ...
           return travel_time_s, takeoff_angle_deg

Then wire it up:

.. code-block:: python

   from hypy.locator import EventLocator
   from hypy.model.layered.tvel import TvelModel

   model = TvelModel("/path/to/model.tvel")
   engine = MyLayeredEngine(model)
   locator = EventLocator(engine, phases=("P", "S"), derivative_method="numerical")

Example: implement the base contract directly
----------------------------------------------

When you need full control (or a non-layered geometry), subclass
``PropagationEngine`` and implement ``compute_travel_time_params`` yourself.
This is the minimal self-contained engine:

.. code-block:: python

   import numpy as np

   from hypy.model import VelocityModel
   from hypy.propagation.base import PropagationEngine


   class MyEngine(PropagationEngine):
       feature_flags = frozenset(
           {
               "propagation.ray",
               "derivative.numerical",
           }
       )
       required_model_features = frozenset({"model.profile.1d"})

       def __init__(self, model: VelocityModel):
           super().__init__(model)

       @property
       def dim(self) -> int:
           return 1

       def compute_travel_time_params(
           self,
           source_latitude,
           source_longitude,
           source_depth_km,
           receiver_latitude,
           receiver_longitude,
           receiver_elevation_m,
           phase,
           delta=None,
           derivative_method="numerical",
       ):
           # Compute these from self.velocity_model and the geometry.
           distance_km = ...
           travel_time_s = ...

           # Return nan derivatives to reject an unusable observation.
           if not np.isfinite(travel_time_s):
               return distance_km, None, (np.nan, np.nan, np.nan, np.nan)

           dt_dr, dt_dx, dt_dy, dt_dz = ...
           return distance_km, travel_time_s, (dt_dr, dt_dx, dt_dy, dt_dz)

See ``examples/custom_backend_contract.py`` for a complete, importable engine
that satisfies this contract.

Common pitfalls
---------------

* **Derivative flag mismatch.** If you request
  ``derivative_method="analytical"`` but the engine only declares
  ``derivative.numerical``, ``EventLocator`` raises ``ValueError`` at setup.
* **Missing model features.** If ``required_model_features`` is not a subset of
  ``model.feature_flags``, the engine constructor raises ``ValueError``.
* **Returning ``None`` derivatives vs. raising.** Reject individual bad
  observations by returning ``None`` / ``nan`` rather than raising — raising
  aborts the whole inversion.
* **Sign of horizontal derivatives.** ``dt_dx`` / ``dt_dy`` are taken with
  respect to **source** coordinates. Moving the source toward the receiver
  reduces distance, so these carry the opposite sign of the path-distance
  derivative. The layered base class handles this for you.
