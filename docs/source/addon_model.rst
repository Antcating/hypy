Adding velocity format
======================

This page documents how to extend ``hypy`` with a custom velocity model backend.

Create a custom model backend
-----------------------------

1. Implement a ``VelocityModel`` subclass.
2. Set ``feature_flags`` to describe capabilities.
3. Register a loader for your file extension.

.. code-block:: python

   from pathlib import Path

   from hypy.model import VelocityModel, register_model_backend


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


   def load_xyz_model(model_path: str | Path, **kwargs) -> VelocityModel:
       del kwargs
       return XYZModel(model_path)


   register_model_backend(".xyz", load_xyz_model)

After registration, any path ending with ``.xyz`` can be loaded through
``create_model`` or ``VelocityModel.from_file``.
