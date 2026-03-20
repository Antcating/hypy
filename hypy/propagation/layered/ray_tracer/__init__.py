from typing import Optional, Union, TYPE_CHECKING

from hypy.propagation.layered import LayeredPropagationModel


class LayeredRayTracer(LayeredPropagationModel):
    """
    Ray tracer for layered Earth models, supporting multiple backends (e.g., Pyrocko, TauP).
    """

    def __init__(self, model_path: str):
        super().__init__(model_path)
