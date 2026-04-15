Adding a custom propagation class
=================================

This page documents how to extend ``hypy`` with a custom propagation class.

Strict constructor and runtime contract
---------------------------------------

When you pass ``engine_backend=YourBackend`` to ``EventLocator``, your class
must satisfy the following contract.

Required class attributes
~~~~~~~~~~~~~~~~~~~~~~~~~

* ``default_feature_flags: frozenset[str]``
* ``default_required_model_features: frozenset[str]``

Required instance members
~~~~~~~~~~~~~~~~~~~~~~~~~

* ``dim`` property returning ``int``
* ``compute_travel_time_params(...)`` with the signature:

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

The return value must be exactly:

* ``distance_km`` as ``float``
* ``travel_time_s`` as ``float``
* derivatives tuple ``(dt_dr, dt_dx, dt_dy, dt_dz)`` as ``float`` values

For invalid paths, return ``numpy.nan`` values instead of ``None`` to keep the
method signature stable while still allowing the locator to reject unusable
observations.

Required constructor compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hypy`` tries these constructor calls in order. Your backend must accept at
least one of them:

1. ``YourBackend(model_path=..., mode=..., engine=..., phases=..., max_depth_km=..., max_distance_km=..., depth_step_km=..., distance_step_km=...)``
2. ``YourBackend(model_path=..., mode=..., engine=..., phases=..., max_depth=..., max_distance=..., depth_step=..., distance_step=...)``
3. ``YourBackend(model_path=...)``

If none matches, setup fails with ``TypeError``.

Required feature semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``default_feature_flags`` must include at least one derivative capability
  used by the locator:

  * ``derivative.numerical`` for ``derivative_method="numerical"``
  * ``derivative.analytical`` for ``derivative_method="analytical"``

* ``default_required_model_features`` must list model capabilities required by
  your backend (for example ``model.profile.1d`` and ``dimension.1d``).
* ``hypy`` validates model/backend compatibility during locator setup and raises
  ``ValueError`` on mismatch.

Recommended for full ``PropagationEngine`` compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods are not currently invoked directly by locator wiring, but are
part of the base class contract and should be implemented for consistency:

* ``get_arrival(...)``
* ``compute_derivatives(...)``

Create a custom propagation class
---------------------------------

.. code-block:: python

   from hypy import EventLocator
   from hypy.propagation.base import PropagationEngine


   class MyRayBackend(PropagationEngine):
       default_feature_flags = frozenset(
           {
               "propagation.layered",
               "propagation.ray",
               "derivative.numerical",
           }
       )
       default_required_model_features = frozenset(
           {
               "model.profile.1d",
               "dimension.1d",
           }
       )

       def __init__(self, model_path: str, **kwargs):
           super().__init__(model_path, mode=kwargs.get("mode"), engine=kwargs.get("engine"))

       @property
       def dim(self) -> int:
           return 1

       def get_arrival(self, *args, **kwargs):
           del args, kwargs
           return 0.0, 0.0, 0.0, 0.0

       def compute_travel_time_params(self, *args, **kwargs):
           del args, kwargs
           return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)


   locator = EventLocator(
       model_path="/path/to/model.tvel",
       mode="ray-1d",
       engine="taup",
       derivative_method="numerical",
       engine_backend=MyRayBackend,
   )
