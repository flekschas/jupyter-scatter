import numpy as np
import numpy.typing as npt
from numba import njit


@njit
def _compute_center_of_mass_jit(hull_points: npt.NDArray[np.float64]):
    """
    JIT-compiled version of center of mass calculation using Shoelace formula
    """

    # Extract x and y coordinates
    x = hull_points[:, 0]
    y = hull_points[:, 1]
    n = len(hull_points)

    # First compute the area using Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j] - x[j] * y[i]
    area = abs(area) / 2.0

    # Then compute centroid
    cx = 0.0
    cy = 0.0

    for i in range(n):
        j = (i + 1) % n
        factor = x[i] * y[j] - x[j] * y[i]
        cx += (x[i] + x[j]) * factor
        cy += (y[i] + y[j]) * factor

    if area > 0:
        cx = cx / (6.0 * area)
        cy = cy / (6.0 * area)
    else:
        # Fallback: simple average if area is too small
        cx = np.mean(x)
        cy = np.mean(y)

    return np.array([cx, cy])


def _compute_center_of_mass_legacy(hull_points: npt.NDArray[np.float64]):
    """
    Legacy version of center of mass calculation using Shoelace formula
    """

    hull_points = np.asarray(hull_points)

    # Ensure hull points are closed (last point == first point)
    if not np.array_equal(hull_points[0], hull_points[-1]):
        hull_points = np.vstack([hull_points, hull_points[0]])

    # Extract x and y coordinates
    x = hull_points[:, 0]
    y = hull_points[:, 1]

    # Compute area using Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Compute centroid coordinates
    cx = np.mean(x)
    cy = np.mean(y)

    # For more accurate center of mass (weighted by area)
    if area > 0:
        cx = (1 / (6 * area)) * np.sum(
            (x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])
        )
        cy = (1 / (6 * area)) * np.sum(
            (y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])
        )

    return np.array([cx, cy])


def compute_center_of_mass(hull_points: npt.NDArray[np.float64]):
    """
    Compute the center of mass of a polygon using the Shoelace formula.

    Parameters
    ----------
    hull_points : np.ndarray
        Array of [x, y] points forming a polygon (convex hull)

    Returns
    -------
    np.ndarray
        Center of mass [x, y]

    Examples
    --------
    >>> points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> compute_center_of_mass(points)
    array([0.5, 0.5])
    """

    try:
        # Try the fast JIT-compiled version first
        return _compute_center_of_mass_jit(hull_points)
    except (TypeError, ValueError, RuntimeError):
        # Fall back to the non-JIT version if that fails
        return _compute_center_of_mass_legacy(hull_points)
