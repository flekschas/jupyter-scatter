from .constants import ASINH_1


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
    denominator = w1 + w2

    if denominator == 0:
        return float('inf')

    return (abs(x1 - x2) * ASINH_1) / denominator
