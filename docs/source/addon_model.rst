Adding a custom velocity model class
====================================

This page documents how to extend ``hypy`` with a custom velocity model class.

Create a custom model class
---------------------------

1. Implement a ``VelocityModel`` subclass.
2. Set ``feature_flags`` to describe capabilities.
3. Pass the class to ``EventLocator(..., velocity_model_backend=...)``.

.. code-block:: python

   from pathlib import Path

    from hypy import EventLocator
    from hypy.model import VelocityModel


   class XYZModel(VelocityModel):
       feature_flags = frozenset(
           {
               "model.profile.1d",
               "dimension.1d",
               "format.xyz",
               "phase.p",
               "phase.s",
           }
       )

       def __init__(self, model_path: str | Path):
           super().__init__(model_path=model_path, dim=1)
           # Load model data here.


     locator = EventLocator(
             model_path="/path/to/model.xyz",
             mode="ray-1d",
             engine="taup",
             derivative_method="numerical",
             velocity_model_backend=XYZModel,
     )

Notes
-----

* Your class constructor must accept ``model_path``.
* ``feature_flags`` should describe dimensionality and data type
    (for example ``model.profile.1d`` or ``model.grid.3d``).
* ``supported_features`` is used during compatibility checks against the
    selected propagation backend.
