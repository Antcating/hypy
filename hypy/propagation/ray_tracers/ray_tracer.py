class RayTracer:
    """
    Unified RayTracer interface to select backend ('pyrocko' or 'taup').
    """
    def __init__(self, model_path: str, engine: str = 'pyrocko'):
        if engine == 'pyrocko':
            from .pyrocko import PyrockoRayTracer
            self.tracer = PyrockoRayTracer(model_path)
        elif engine == 'taup':
            from .taup import TauPTracer
            self.tracer = TauPTracer(model_path)
        else:
            raise ValueError(f"Unknown backend: {engine}. Choose 'pyrocko' or 'taup'.")

    def get_arrival(self, source_depth: float, distance: float, phase: str):
        return self.tracer.get_arrival(source_depth, distance, phase)
