from .pyrocko import PyrockoRayTracer
from .taup import TauPTracer

class RayTracer:
    """
    Unified RayTracer interface to select backend ('pyrocko' or 'taup').
    """
    def __init__(self, model_path: str, backend: str = 'pyrocko'):
        if backend == 'pyrocko':
            self.tracer = PyrockoRayTracer(model_path)
        elif backend == 'taup':
            self.tracer = TauPTracer(model_path)
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'pyrocko' or 'taup'.")

    def get_arrival(self, source_depth: float, distance: float, phase: str):
        return self.tracer.get_arrival(source_depth, distance, phase)
