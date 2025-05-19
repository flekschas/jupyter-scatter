import numpy as np
import numpy.typing as npt


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

    hull_points = np.asarray(hull_points)

    # Ensure hull points are closed (last point == first point)
    if not np.array_equal(hull_points[0], hull_points[-1]):
        hull_points = np.vstack([hull_points, hull_points[0]])

    # Extract x and y coordinates
    x = hull_points[:, 0]
    y = hull_points[:, 1]

    # Compute area using Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Compute centroid coordinates (excluding the duplicated first point)
    cx = np.mean(x[:-1])
    cy = np.mean(y[:-1])

    # For more accurate center of mass (weighted by area)
    if area > 0:
        cx = (1 / (6 * area)) * np.sum(
            (x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])
        )
        cy = (1 / (6 * area)) * np.sum(
            (y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])
        )

    return np.array([cx, cy])
