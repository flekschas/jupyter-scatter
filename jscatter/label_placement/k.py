from numba import jit

from .constants import ASINH_1


@jit(nopython=True)
def _compute_k_jit(x1: float, w1: float, x2: float, w2: float):
    """JIT-compiled version of constant 'k' calculation"""
    denominator = w1 + w2
    if denominator == 0:
        return float('inf')
    return (abs(x1 - x2) * ASINH_1) / denominator


def _compute_k_legacy(x1: float, w1: float, x2: float, w2: float):
    """Legacy version of constant 'k' calculation"""
    denominator = w1 + w2
    if denominator == 0:
        return float('inf')
    return (abs(x1 - x2) * ASINH_1) / denominator


def compute_k(x1: float, w1: float, x2: float, w2: float):
    """
    Get x-based constant 'k' for collision detection.

    Parameters
    ----------
    x1 : float
        Center x/y of first label
    w1 : float
        Half width/height of first label
    x2 : float
        Center x/y of second label
    w2 : float
        Half width/height of second label

    Returns
    -------
    float
        k value for zoom solver
    """
    try:
        # Try the fast JIT-compiled version first
        return _compute_k_jit(x1, w1, x2, w2)
    except (TypeError, ValueError, RuntimeError):
        # Fall back to the non-JIT version if that fails
        return _compute_k_legacy(x1, w1, x2, w2)
