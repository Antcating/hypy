"""Finite-difference derivative helpers for scalar functions."""

from typing import Callable


def finite_difference_derivative(func: Callable, x: float, h: float, method: str):
    """Compute a finite-difference derivative of a scalar function.

    Parameters
    ----------
    func : callable
        Function to differentiate. It must accept a single float argument and
        return a scalar value or ``None``.
    x : float
        Point at which to evaluate the derivative.
    h : float
        Finite-difference step size.
    method : str
        Differencing scheme to use. Supported values are ``"forward"``,
        ``"backward"``, ``"central"``, and ``"five-point"``.

    Returns
    -------
    float
        Estimated derivative at ``x``.

    Raises
    ------
    ValueError
        If the function returns ``None`` for required samples or if ``method``
        is not recognized.

    Notes
    -----
    The central and five-point schemes degrade gracefully near boundaries by
    falling back to lower-order formulas when one-sided samples are available.
    """

    def _value(at_x: float):
        result = func(at_x)
        if result is None:
            raise ValueError(f"Function returned None for derivative at x={at_x}.")
        return result

    if method == "forward":
        return (_value(x + h) - _value(x)) / h
    elif method == "backward":
        return (_value(x) - _value(x - h)) / h
    elif method == "central":
        t_plus = func(x + h)
        t_minus = func(x - h)
        t_zero = func(x)

        if t_plus is not None and t_minus is not None:
            return (t_plus - t_minus) / (2 * h)
        if t_plus is not None and t_zero is not None:
            return (t_plus - t_zero) / h
        if t_minus is not None and t_zero is not None:
            return (t_zero - t_minus) / h

        raise ValueError(
            f"Unable to compute central finite difference derivative at x={x}: "
            "function returned None for required sample points."
        )
    elif method == "five-point":
        t_plus_2 = func(x + 2 * h)
        t_plus = func(x + h)
        t_minus = func(x - h)
        t_minus_2 = func(x - 2 * h)

        if (
            t_plus_2 is not None
            and t_plus is not None
            and t_minus is not None
            and t_minus_2 is not None
        ):
            return (-t_plus_2 + 8 * t_plus - 8 * t_minus + t_minus_2) / (12 * h)

        return finite_difference_derivative(func, x, h, "central")
    else:
        raise ValueError(
            "Invalid stencil type. Choose from 'forward', 'backward', "
            "'central', 'five-point'."
        )
