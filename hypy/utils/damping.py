from typing import Optional

import numpy as np


def update_damping_factor(
    current_factor: float,
    rms_prev: float,
    rms_new: float,
    accepted: Optional[bool] = None,
    dec: float = 1 / 3,  # how much to decrease when things go well (λ ← λ * dec)
    inc: float = 2.0,  # how much to increase when things go badly (λ ← λ * inc)
    min_factor: float = 1e-6,
    max_factor: float = 1e6,
) -> float:
    """
    Adaptive damping:
      - If the step reduced RMS (or was accepted), decrease damping (toward Gauss-Newton).
      - If the step didn't help, increase damping (toward gradient descent).
    """
    if accepted is None:
        accepted = rms_new < rms_prev * (1 - 1e-4)  # small improvement threshold

    new_lambda = current_factor * (dec if accepted else inc)
    return float(min(max(new_lambda, min_factor), max_factor))
