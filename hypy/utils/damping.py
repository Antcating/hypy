"""Utilities for adjusting damping during iterative inversion."""

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
    """Update the Levenberg-Marquardt damping factor.

    Parameters
    ----------
    current_factor : float
        Current damping factor.
    rms_prev : float
        Residual root-mean-square before the candidate update.
    rms_new : float
        Residual root-mean-square after the candidate update.
    accepted : bool or None, optional
        Whether the update was accepted. If ``None``, acceptance is inferred
        from ``rms_new < rms_prev``.
    dec : float, optional
        Multiplicative factor used to decrease the damping term when the
        update is accepted.
    inc : float, optional
        Multiplicative factor used to increase the damping term when the
        update is rejected.
    min_factor : float, optional
        Lower bound applied to the updated damping factor.
    max_factor : float, optional
        Upper bound applied to the updated damping factor.

    Returns
    -------
    float
        The updated damping factor clipped to ``[min_factor, max_factor]``.

    Notes
    -----
    The factor is scaled rather than recomputed so the caller can preserve the
    trust-region history across iterations. The clipping step prevents the
    solver from drifting into numerically unstable damping regimes.
    """
    if accepted is None:
        accepted = rms_new < rms_prev

    new_lambda = current_factor * (dec if accepted else inc)
    return float(np.clip(new_lambda, min_factor, max_factor))
