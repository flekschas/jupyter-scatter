import numpy as np
import numpy.typing as npt
from numba import njit


@njit
def _compute_highest_density_point_jit(
    points: npt.NDArray[np.float64],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    grid_size: int = 50,
):
    """
    JIT-compiled version of highest density point calculation
    """

    # Initialize histogram
    hist = np.zeros((grid_size, grid_size))

    # Populate histogram
    for i in range(len(points)):
        x, y = points[i]
        x_idx = min(int((x - x_min) / (x_max - x_min) * grid_size), grid_size - 1)
        y_idx = min(int((y - y_min) / (y_max - y_min) * grid_size), grid_size - 1)
        hist[x_idx, y_idx] += 1

    # Find the bin with the highest count
    max_count = 0
    max_x_idx = 0
    max_y_idx = 0

    for i in range(grid_size):
        for j in range(grid_size):
            if hist[i, j] > max_count:
                max_count = hist[i, j]
                max_x_idx = i
                max_y_idx = j

    # Get the center coordinates of that bin
    x_center = x_min + (max_x_idx + 0.5) * (x_max - x_min) / grid_size
    y_center = y_min + (max_y_idx + 0.5) * (y_max - y_min) / grid_size

    return np.array([x_center, y_center])


def _compute_highest_density_point_legacy(
    points: npt.NDArray[np.float64],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    grid_size: int = 50,
):
    """
    Legacy version of highest density point calculation using numpy's histogram2d
    """
    if len(points) == 0:
        return np.array([0, 0])

    if len(points) <= 2:
        # For 1-2 points, just return the mean
        return np.mean(points, axis=0)

    # Add a small buffer to ensure all points are included
    x_range = [x_min, x_max]
    y_range = [y_min, y_max]

    points = np.asarray(points)

    # Create the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=grid_size,
        range=[x_range, y_range],
    )

    # Find the bin with the highest count
    max_idx = np.unravel_index(np.argmax(hist), hist.shape)

    # Get the center coordinates of that bin
    x_center = (x_edges[max_idx[0]] + x_edges[max_idx[0] + 1]) / 2
    y_center = (y_edges[max_idx[1]] + y_edges[max_idx[1] + 1]) / 2

    return np.array([x_center, y_center])


def compute_highest_density_point(points: npt.NDArray[np.float64], grid_size: int = 50):
    """
    Compute the point of highest density using a 2D histogram.

    Parameters
    ----------
    points : numpy.ndarray
        Array of [x, y] points
    grid_size : int, default=50
        Number of bins in each dimension for the histogram

    Returns
    -------
    highest_density_point : numpy.ndarray
        [x, y] coordinates of the highest density point

    Examples
    --------
    >>> points = np.array([[0, 0], [0.1, 0.1], [0.9, 0.9], [1, 1]])
    >>> compute_highest_density_point(points)
    array([0.05, 0.05])  # Approximate result
    """
    if len(points) == 0:
        return np.array([0, 0])

    if len(points) <= 2:
        # For 1-2 points, just return the mean
        return points[0]

    # Calculate the range for the histogram
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    try:
        # Try the fast JIT-compiled version first
        return _compute_highest_density_point_jit(
            points, x_min, x_max, y_min, y_max, grid_size
        )
    except (TypeError, ValueError, RuntimeError):
        # Fall back to the numpy version if that fails
        return _compute_highest_density_point_legacy(
            points, x_min, x_max, y_min, y_max, grid_size
        )
