Adding propagation backend
==========================

This page documents how to extend ``hypy`` with a custom propagation backend.

Create a custom propagation backend
-----------------------------------

1. Implement a ``PropagationEngine`` subclass.
2. Return your backend from a factory.
3. Register it for a ``(mode, engine)`` pair.

.. code-block:: python

   from pathlib import Path

   from hypy.propagation import register_propagation_backend
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

       @property
       def dim(self) -> int:
           return 1

       def get_arrival(self, *args, **kwargs):
           del args, kwargs
           return 0.0, 0.0, 0.0, 0.0

       def compute_travel_time_params(self, *args, **kwargs):
           del args, kwargs
           return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)


   def create_my_ray_backend(*, model_path: Path, mode: str, engine: str, **kwargs):
       del kwargs
       return MyRayBackend(model_path=model_path.as_posix(), mode=mode, engine=engine)


   register_propagation_backend("ray-1d", "myengine", create_my_ray_backend)
