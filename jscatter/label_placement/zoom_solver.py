import math

from numba import jit
from scipy.optimize import brentq

# Constants
ASINH_1 = math.asinh(1)

# Region thresholds
REGION_1_THRESHOLD = 0.87
REGION_2_THRESHOLD = 0.7
REGION_3_THRESHOLD = 0.4
REGION_4_THRESHOLD = 0.1

# Region-specific constants
REGION_1_SLOPE = 8.0
REGION_2_LINEAR = 5.0
REGION_2_QUADRATIC = 10.0
REGION_3_NUMERATOR = 4.7
REGION_3_DENOMINATOR = 0.94
REGION_4_INVERSE = 1.45
REGION_4_OFFSET = 5.0
REGION_5_INVERSE = 1.49

# Numerical stability limits
SMALLEST_SAFE_K = 1e-10
MAX_SAFE_ZOOM = 1e9


@jit(nopython=True)
def _f(z: float, k: float):
    """
    JIT-compiled inverse hyperbolic zoom scale function for which we need to find the root.
    The root represents the zoom level at which overlaps are resolved.

    Args:
        z: Zoom scale between [1, Infinity]
        k: Constant representing Math.abs(x1 - x2) * 2 * ASINH_1 / (w1 + w2)

    Returns:
        Function value at point z
    """
    return math.asinh(z) / z - k


# Non-JIT version for fallback
def _f_legacy(z: float, k: float):
    """
    Inverse hyperbolic zoom scale function for which we need to find the root.
    The root represents the zoom level at which overlaps are resolved.

    Args:
        z: Zoom scale between [1, Infinity]
        k: Constant representing Math.abs(x1 - x2) * 2 * ASINH_1 / (w1 + w2)

    Returns:
        Function value at point z
    """
    return math.asinh(z) / z - k


def _safe_brentq(k: float, a: float, b: float, rtol: float):
    try:
        # Try with JIT compilation first
        return brentq(lambda z: _f(z, k), a, b, rtol=rtol)
    except (TypeError, ValueError, RuntimeError):
        # If that fails, try the regular method
        return brentq(lambda z: _f_legacy(z, k), a, b, rtol=rtol)


def _find_search_interval_with_grid(
    k: float, min_val: float = 1.0, max_val: float = 1000.0, points: int = 50
):
    """
    Helper function to find a search interval with grid search.

    Args:
        k: Constant value
        min_val: Minimum value for grid search
        max_val: Maximum value for grid search
        points: Number of points to evaluate

    Returns:
        Tuple containing the bracket [a, b] or None if no bracket is found
    """
    # Use a logarithmic grid for better coverage
    log_min = math.log(min_val)
    log_max = math.log(max_val)
    step = (log_max - log_min) / (points - 1)

    for i in range(points - 1):
        a = math.exp(log_min + i * step)
        b = math.exp(log_min + (i + 1) * step)
        fa = _f_legacy(a, k)
        fb = _f_legacy(b, k)

        if fa * fb <= 0:
            return a, b

    return None


def _find_search_interval(
    k: float, start: float = 1.0, initial_step: float = 1.0, max_iterations: int = 100
):
    """
    Function to find search interval with adaptive step size.

    Args:
        k: Constant value
        start: Starting point
        initial_step: Initial step size
        max_iterations: Maximum number of iterations

    Returns:
        Tuple containing the bracket [a, b] or None if no bracket is found
    """
    a = start
    fa = _f_legacy(a, k)
    step = initial_step

    for i in range(max_iterations):
        # Try in positive direction
        b = a + step
        fb = _f_legacy(b, k)

        if fa * fb <= 0:
            return a, b

        # Try in negative direction (if possible without going below 1.0)
        if a - step >= 1.0:
            b = a - step
            fb = _f_legacy(b, k)

            if fa * fb <= 0:
                return b, a

        # Increase step size if we haven't found a bracket
        step *= 2

        # If step becomes too large, try a more systematic approach
        if step > 1000:
            return _find_search_interval_with_grid(k)

    # If we failed with adaptive step sizing, try grid search
    return _find_search_interval_with_grid(k)


def solve_zoom_approximately(k: float):
    """
    Fast approximation function for solving zoom level
    using a piecewise model.

    Args:
        k: Constant value

    Returns:
        Approximate zoom level
    """
    # Error handling for invalid input
    if not isinstance(k, (int, float)) or math.isnan(k):
        raise TypeError('Input k must be a number')

    if k <= 0:
        raise ValueError(f'Input k must be positive; got {k}')

    if k >= ASINH_1:
        return 1.0

    # Handle extreme cases for numerical stability
    if k < SMALLEST_SAFE_K:
        return min(REGION_5_INVERSE / SMALLEST_SAFE_K, MAX_SAFE_ZOOM)

    # Piecewise approximation based on k range
    if k > REGION_1_THRESHOLD:
        return 1 + REGION_1_SLOPE * (ASINH_1 - k)

    if k >= REGION_2_THRESHOLD:
        delta = ASINH_1 - k
        return 1 + REGION_2_LINEAR * delta + REGION_2_QUADRATIC * delta * delta

    if k >= REGION_3_THRESHOLD:
        t = 1 - k / ASINH_1
        return 1 + (REGION_3_NUMERATOR * t) / (1 - REGION_3_DENOMINATOR * t)

    if k >= REGION_4_THRESHOLD:
        return REGION_4_INVERSE / k + REGION_4_OFFSET

    return REGION_5_INVERSE / k


def solve_zoom_precisely(k: float, tolerance: float = 1e-10):
    """
    Precise numerical solution for zoom levels using scipy's brentq method.

    Args:
        k: Constant value
        tolerance: Error tolerance

    Returns:
        Precise zoom level
    """
    # Handle boundary cases
    if k >= ASINH_1:
        return 1.0

    if k <= 0:
        raise ValueError(f'Input k must be positive; got {k}')

    interval = _find_search_interval(k)
    if interval is None:
        # Try grid search as a fallback
        interval = _find_search_interval_with_grid(k)
        if interval is None:
            raise ValueError(f'Could not find search interval for k = {k}')

    a, b = interval

    try:
        return _safe_brentq(k, a, b, rtol=tolerance)
    except:
        # If the root finding method fails, use the approximation
        return solve_zoom_approximately(k)


def solve_zoom(k: float, tolerance: float = 1e-10):
    """
    Hybrid solution for solving zoom level, starting with approximation
    and refining if needed.

    Args:
        k: Constant value
        tolerance: Error tolerance

    Returns:
        Optimal zoom level
    """
    # Handle boundary cases
    if k >= ASINH_1:
        return 1.0
    if k <= 0:
        raise ValueError('Invalid k value')

    # Try approximation first
    approx = solve_zoom_approximately(k)

    # Test how good the approximation is
    try:
        error = abs(math.asinh(approx) / approx - k)
    except:
        # If calculation fails, just use the approximation
        return approx

    # If approximation is good enough, use it
    if error < tolerance:
        return approx

    # Use the approximation to construct a likely search interval
    a = max(1.0, approx * 0.5)
    b = approx * 2.0

    try:
        return _safe_brentq(k, a, b, rtol=tolerance)
    except:
        # If the bracket fails, use the more robust interval finding
        interval = _find_search_interval(k)

        if interval is None:
            # If we still can't find a search interval, the approximation
            # is our best bet
            return approx

        a, b = interval

        return _safe_brentq(k, a, b, rtol=tolerance)
