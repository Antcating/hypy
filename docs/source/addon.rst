Addon Example
=============

This page shows how to add a small extension, or addon, to ``hypy`` without
modifying the core package. The current architecture supports two practical
extension points:

* propagation backends through the ``hypy.propagation.backends`` entry-point group
* model loaders through ``VelocityModel.register_loader``

Add a propagation addon
-----------------------

A propagation addon is the recommended extension path when you want to add a
new ray-tracer or travel-time backend.

Example package layout:

.. code-block:: text

   my_package/
     __init__.py
    backends.py
     pyproject.toml

Example addon factory:

.. code-block:: python

   from pathlib import Path

   from hypy.propagation.base import PropagationModel


    class MyBackend(PropagationModel):
       def __init__(self, model_path: str):
           super().__init__(model_path)

       @property
       def dim(self) -> int:
           return 3

       def get_arrival(self, *args, **kwargs):
           return 0.0, None, None, 0.0

       def compute_travel_time_params(self, *args, **kwargs):
           return 0.0, 0.0, (None, None, None, None)


   def create_my_backend(*, model_path: Path, **kwargs) -> PropagationModel:
       del kwargs
       return MyBackend(model_path.as_posix())

Register the addon through ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."hypy.propagation.backends"]
    "table-3d:fmm" = "my_package.backends:create_my_backend"

Then load plugins before creating propagation backends:

.. code-block:: python

    from hypy.propagation import create_propagation_backend, load_propagation_plugins

    load_propagation_plugins()
    backend = create_propagation_backend(
       mode="table-3d",
       engine="fmm",
       phases=("P",),
       model_path="/path/to/model.nc",
       max_depth=50.0,
       max_distance=100.0,
       depth_step=1.0,
       distance_step=1.0,
   )

Add a model loader addon
------------------------

If you only need to teach ``hypy`` how to read a new model file format, register a
loader with ``VelocityModel`` at application startup.

.. code-block:: python

   from hypy.model import VelocityModel


   def load_xyz_model(path, **kwargs):
       del kwargs
       ...  # return a VelocityModel subclass


   VelocityModel.register_loader(".xyz", load_xyz_model)

This is useful when your addon produces a model wrapper rather than a ray-tracer
backend.

Recommended pattern
-------------------

1. Keep addon code outside the core ``hypy`` package.
2. Register the addon once during app startup.
3. Prefer the public package roots and registry APIs over deep imports.
4. Add tests that import the addon and exercise the registration path.
