class PropagationModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def get_arrival(self, source_depth, distance, phase):
        # Placeholder for child classes to implement 
        # arrival time and takeoff angle retrieval
        raise NotImplementedError("This method should be overridden by subclasses")
    
