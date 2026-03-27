import numpy as np


def distance_weight(distance, **kwargs):
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

    # https://www.desmos.com/calculator/221eudc8kc
    weight = 1.0 / (1.0 + np.exp(k * (distance - x0)))

    return weight


def time_error_weight(time_error, **kwargs):
    """
    Compute time error weight based on logistic function.
    Parameters
    ----------
    time_error : float or np.ndarray
        Time error in seconds.
    """

    weight = 1.0 / ((time_error + 1) ** 2)

    return weight
