import numpy as np
import numpy.typing as npt


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

    if len(points) == 1:
        # For 1 point, just return the point
        return points[0]

    if len(points) == 2:
        # For 2 points, just return the mean
        return np.mean(points, axis=0)

    # Calculate the range for the histogram
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    points = np.asarray(points)

    # Create the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=grid_size,
        range=[[x_min, x_max], [y_min, y_max]],
    )

    # Find the bin with the highest count
    max_idx = np.unravel_index(np.argmax(hist), hist.shape)

    # Get the center coordinates of that bin
    x_center = (x_edges[max_idx[0]] + x_edges[max_idx[0] + 1]) / 2
    y_center = (y_edges[max_idx[1]] + y_edges[max_idx[1] + 1]) / 2

    return np.array([x_center, y_center])
