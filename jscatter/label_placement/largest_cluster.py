import warnings

import numpy as np
import numpy.typing as npt

from ..dependencies import check_label_extras_dependencies

# Filter HDBSCAN warnings
warnings.filterwarnings(
    'ignore', category=SyntaxWarning, message='invalid escape sequence.*'
)
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message="'force_all_finite' was renamed to 'ensure_all_finite'.*",
)


def compute_largest_cluster(points: npt.NDArray[np.float64], max_points: int = 1000):
    """
    Compute the points of the largest cluster using apricot downsampling + fast_hdbscan.

    Parameters
    ----------
    points : numpy.ndarray
        Array of [x, y] points
    max_points : int, default=1000
        Maximum number of points to consider

    Returns
    -------
    largest_cluster_points : numpy.ndarray
        [x, y] coordinates of the points of the largest cluster
    """
    # Ensure required dependencies are installed
    check_label_extras_dependencies()

    from hdbscan import HDBSCAN

    if points.size == 0:
        return np.array([0, 0]).reshape((1, 2))

    if len(points) <= 3:
        # For very few points, just return the mean
        return np.mean(points, axis=0).reshape((1, 2))

    # Only downsample if we have more points than our threshold
    if len(points) > max_points:
        # Random subsample
        chosen_idxs = np.random.choice(len(points), size=max_points, replace=False)
        downsampled_points = points[chosen_idxs]
    else:
        downsampled_points = points

    # Run fast_hdbscan on the downsampled points
    # fast_hdbscan works very well for 2D data
    min_cluster_size = max(5, len(downsampled_points) // 20)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(downsampled_points)

    # Noise points are labeled as `-1`
    valid_labels = cluster_labels[cluster_labels != -1]

    if len(valid_labels) == 0:
        # If all points are noise, return mean of all points
        return np.mean(downsampled_points, axis=0).reshape((1, 2))

    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    largest_cluster_idx = unique_labels[np.argmax(counts)]

    # Return points from the largest cluster
    return downsampled_points[cluster_labels == largest_cluster_idx].reshape((-1, 2))
