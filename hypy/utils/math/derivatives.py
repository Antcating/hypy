from typing import Callable


def finite_difference_derivative(func: Callable, x: float, h: float, method: str):
    """
    Compute the numerical derivative of a function at a point using a specified finite difference stencil.

    Parameters
    ----------
    func : callable
        The function for which to compute the derivative. Should take a single argument.
    x : float
        The point at which to compute the derivative.
    h : float
        The step size for finite differences.
    stencil : str
        The type of finite difference stencil to use. Options are 'forward', 'backward', 'central', 'five-point'.

    Returns
    -------
    float
        The numerical derivative of func at point x.
    """
    if method == "forward":
        return (func(x + h) - func(x)) / h
    elif method == "backward":
        return (func(x) - func(x - h)) / h
    elif method == "central":
        return (func(x + h) - func(x - h)) / (2 * h)
    elif method == "five-point":
        return (
            -func(x + 2 * h) + 8 * func(x + h) - 8 * func(x - h) + func(x - 2 * h)
        ) / (12 * h)
    else:
        raise ValueError(
            "Invalid stencil type. Choose from 'forward', 'backward', 'central', 'five-point'."
        )
