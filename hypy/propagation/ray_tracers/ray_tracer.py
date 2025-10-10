from ..propagation_model import PropagationModel


class RayTracer(PropagationModel):
    """
    Unified RayTracer interface to select backend ('pyrocko' or 'taup').
    """

    def __init__(self, model_path: str, engine: str = "pyrocko"):
        self.tracer: PropagationModel
        if engine == "pyrocko":
            from .pyrocko import PyrockoRayTracer

            self.tracer = PyrockoRayTracer(model_path)
        elif engine == "taup":
            from .taup import TauPRayTracer

            self.tracer = TauPRayTracer(model_path)
        else:
            raise ValueError(f"Unknown backend: {engine}. Choose 'pyrocko' or 'taup'.")

    def get_arrival(self, source_depth: float, distance: float, phase: str):
        return self.tracer.get_arrival(source_depth, distance, phase)
