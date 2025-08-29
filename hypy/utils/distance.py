import numpy as np

def compute_distance_weight(distance, **kwargs):
    """
    Compute distance weight based on logistic function.
    Parameters
    ----------
    distance : float or np.ndarray
        Distance in kilometers.
    """

    k = kwargs.get("k", None)
    x0 = kwargs.get("x0", None)

    if k is None or x0 is None:
        raise ValueError("Parameters 'k' and 'x0' must be provided in kwargs.")

    weight = 1.0 / (1.0 + np.exp(k * (distance - x0)))

    return weight