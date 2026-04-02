from typing import Optional

import numpy as np


def update_damping_factor(
    current_factor: float,
    rms_prev: float,
    rms_new: float,
    accepted: Optional[bool] = None,
    dec: float = 0.5,
    inc: float = 5.0,
    min_factor: float = 1e-8,
    max_factor: float = 1e6,
) -> float:
    if accepted is None:
        accepted = rms_new < rms_prev

    new_lambda = current_factor * (dec if accepted else inc)
    return float(np.clip(new_lambda, min_factor, max_factor))
