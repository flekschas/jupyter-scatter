import numpy as np
import numpy.typing as npt

from ..types import AggregationMethod


def aggregate(
    values: npt.NDArray[np.float64], method: AggregationMethod = 'mean'
) -> np.float64:
    """
    Aggregate importance values using the specified method.

    Parameters
    ----------
    values : npt.NDArray[np.float64]
        Array of importance values to aggregate
    method : ImportanceAggregator
        Aggregation method, either a string ('min', 'max', 'median', 'mean', 'sum')
        or a callable that takes a numpy array and returns any numeric value
        that can be cast to np.float64

    Returns
    -------
    np.float64
        Aggregated importance value

    Examples
    --------
    >>> values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> aggregate_importance(values, 'mean')
    3.0
    >>> aggregate_importance(values, 'max')
    5.0
    >>> aggregate_importance(values, lambda x: np.percentile(x, 75))
    4.0
    >>> aggregate_importance(values, lambda x: int(np.sum(x)))  # Returns int cast to float64
    15.0
    """
    if len(values) == 0:
        return np.float64(0.0)

    if callable(method):
        # Cast the result to np.float64 regardless of its original type
        return np.float64(method(values))

    if method == 'min':
        return np.float64(np.min(values))
    elif method == 'max':
        return np.float64(np.max(values))
    elif method == 'median':
        return np.float64(np.median(values))
    elif method == 'mean':
        return np.float64(np.mean(values))
    elif method == 'sum':
        return np.float64(np.sum(values))
    else:
        raise ValueError(
            f'Unknown aggregation method: {method}. '
            f"Expected one of 'min', 'max', 'median', 'mean', 'sum' "
            f'or a callable.'
        )
