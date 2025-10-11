from pathlib import Path


class PropagationModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = Path(model_path).stem
        self.model = None

    def get_arrival(self, source_depth, distance, phase):
        # Placeholder for child classes to implement
        # arrival time and takeoff angle retrieval
        raise NotImplementedError("This method should be overridden by subclasses")
