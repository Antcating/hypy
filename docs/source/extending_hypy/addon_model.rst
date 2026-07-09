Adding a custom velocity model
==============================

A velocity model wraps the Earth-structure data that a propagation engine
needs. Custom models let you support new file formats or data sources while
remaining interchangeable with the built-in models.

All models subclass :class:`hypy.model.VelocityModel`.

The contract
------------

Your subclass must:

1. Declare a class attribute ``feature_flags: frozenset[str]`` describing the
   model's capabilities (dimensionality, format, supported phases).
2. Accept a ``model_path`` in ``__init__`` and call ``super().__init__(...)``
   with at least ``model_path`` and ``dim`` (the spatial dimensionality).
3. Load and expose whatever data your propagation engine will query.

The base class provides the compatibility surface used during setup:

* ``feature_flags`` — declared by you (class attribute).
* ``supported_features`` — ``feature_flags`` plus inferred flags such as
  ``dimension.1d`` (derived from ``dim``).
* ``supports_feature(...)`` / ``supports_all_features(...)`` — convenience
  checks used by engines and the locator.

Minimal example
---------------

.. code-block:: python

   from pathlib import Path

   import numpy as np

   from hypy.model import VelocityModel


   class XYZModel(VelocityModel):
       """1D velocity model loaded from a simple two-column text file."""

       feature_flags = frozenset(
           {
               "model.profile.1d",
               "model.layered",
               "format.xyz",
               "phase.p",
               "phase.s",
           }
       )

       def __init__(self, model_path: str | Path):
           model_path = Path(model_path).expanduser().resolve()
           if not model_path.exists():
               raise FileNotFoundError(f"Model file not found: {model_path}")

           # dim=1 marks this as a 1D profile model.
           super().__init__(model_path=model_path, model_name=model_path.stem, dim=1)

           # Load model data into whatever structure your engine expects.
           data = np.loadtxt(model_path)
           self._depth_km = data[:, 0]
           self._vp_km_s = data[:, 1]
           self._vs_km_s = data[:, 2]

       def get_velocity_1d(self, depth_km: float, phase: str) -> float:
           """Return interpolated velocity (km/s) at depth for a phase."""
           velocity = self._vp_km_s if phase.upper().startswith("P") else self._vs_km_s
           return float(np.interp(depth_km, self._depth_km, velocity))

Using the model
---------------

Instantiate the model and hand it to a propagation engine, then to the
locator:

.. code-block:: python

   from hypy.locator import EventLocator
   from hypy.propagation.layered.ray_tracer.hypo import HypoRayTracer

   model = XYZModel("/path/to/model.xyz")
   engine = HypoRayTracer(model)
   locator = EventLocator(engine, phases=("P", "S"))

What the engine expects from the model
--------------------------------------

The base :class:`~hypy.model.VelocityModel` only standardizes metadata
(``model_path``, ``model_name``, ``dim``, ``radius_of_planet_km``) and the
feature-flag surface. The **data access methods** a model must provide are
dictated by the engine you pair it with, not by the base class.

For example, the built-in 1D engines call
:meth:`~hypy.model.layered.tvel.TvelModel.get_velocity_1d` (for analytical
derivatives) and the layer-profile helpers. If you write a custom model for an
existing engine, mirror the method names that engine calls. If you write both a
custom model and a custom engine (see :doc:`addon_propagation`), you are free
to define your own access methods.

Notes
-----

* Your constructor **must** accept ``model_path`` (positional). This keeps
  models loadable through :meth:`~hypy.model.VelocityModel.from_file` routing
  when the suffix is recognized.
* ``feature_flags`` should describe dimensionality and data type. Use
  ``model.profile.1d`` for 1D profiles and ``model.grid.3d`` for 3D grids; the
  base class adds ``dimension.<dim>d`` automatically.
* ``feature_flags`` is compared against each engine's
  ``required_model_features`` during engine construction. Make sure your flags
  cover everything the target engine requires, or construction raises
  ``ValueError``.
* To make a new file suffix loadable through ``VelocityModel.from_file(path)``,
  extend ``supported_model_file_types`` / ``_resolve_builtin_loader`` in
  :mod:`hypy.model`. Direct instantiation (``XYZModel(path)``) works without
  this step.
