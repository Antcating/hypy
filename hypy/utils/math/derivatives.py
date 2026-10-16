"""Finite-difference derivative helpers for scalar and batched functions."""

from typing import Callable


def finite_difference_derivative(
    func: Callable,
    x: float,
    h: float,
    method: str,
):
    """Compute a finite-difference derivative of a scalar function."""

    def _value(at_x: float):
        result = func(at_x)
        if result is None:
            raise ValueError(f"Function returned None for derivative at x={at_x}.")
        return result

    if method == "forward":
        return (_value(x + h) - _value(x)) / h

    if method == "central":
        t_plus = func(x + h)
        t_minus = func(x - h)
        t_zero = func(x)

        if t_plus is not None and t_minus is not None:
            return (t_plus - t_minus) / (2.0 * h)

        if t_plus is not None and t_zero is not None:
            return (t_plus - t_zero) / h

        if t_minus is not None and t_zero is not None:
            return (t_zero - t_minus) / h

        raise ValueError(
            f"Unable to compute central finite difference derivative at x={x}: "
            "function returned None for required sample points."
        )

    raise ValueError("Invalid stencil type. Choose from 'forward', 'central'.")
