import pytest


def test_aggregate_function():
    """Test the aggregate function with various methods."""
    import numpy as np

    from jscatter.label_placement.aggregate import aggregate

    # Create test data
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Test different aggregation methods
    assert aggregate(values, 'min') == 1.0
    assert aggregate(values, 'max') == 5.0
    assert aggregate(values, 'median') == 3.0
    assert aggregate(values, 'mean') == 3.0
    assert aggregate(values, 'sum') == 15.0

    # Test with custom callable
    assert aggregate(values, lambda x: np.percentile(x, 75)) == 4.0

    # Test with empty array
    assert aggregate(np.array([]), 'mean') == 0.0

    # Test with invalid method
    with pytest.raises(ValueError):
        aggregate(values, 'invalid_method')


def test_importance_aggregation_methods():
    """Test that different importance aggregation methods affect label priority correctly."""
    import numpy as np
    import pandas as pd

    from jscatter.label_placement import LabelPlacement

    # Create a simple dataset with multiple points per label
    df = pd.DataFrame(
        {
            'x': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'y': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'labelA': ['A', 'A', 'A', 'B', 'B', 'C'],
            'labelB': ['A1', 'A2', 'A3', 'B1', 'B2', 'C1'],
            'importance': [1, 10, 3, 7, 2, 4],
        }
    )

    # Test with default aggregation method (mean)
    label_placer_mean = LabelPlacement(
        data=df,
        by='labelA',
        x='x',
        y='y',
        importance='importance',
        importance_aggregation='mean',  # This is the default
    )

    labels_mean = label_placer_mean.compute()

    # With mean aggregation:
    # A: mean([1, 10, 3]) = 4.67
    # B: mean([7, 2]) = 4.5
    # C: mean([4]) = 4
    # So the order should be C, A, B
    assert labels_mean.iloc[0]['label'] == 'A'
    assert labels_mean.iloc[1]['label'] == 'B'
    assert labels_mean.iloc[2]['label'] == 'C'

    # Test with sum aggregation
    label_placer_sum = LabelPlacement(
        data=df,
        by='labelA',
        x='x',
        y='y',
        importance='importance',
        importance_aggregation='sum',
    )

    labels_sum = label_placer_sum.compute()

    # With sum aggregation:
    # A: sum([1, 10, 3]) = 14
    # B: sum([7, 2]) = 9
    # C: sum([5]) = 5
    # So the order should be A, B, C
    assert labels_sum.iloc[0]['label'] == 'A'
    assert labels_sum.iloc[1]['label'] == 'B'
    assert labels_sum.iloc[2]['label'] == 'C'

    # Test with max aggregation
    label_placer_max = LabelPlacement(
        data=df,
        by='labelA',
        x='x',
        y='y',
        importance='importance',
        importance_aggregation='max',
    )

    labels_max = label_placer_max.compute()

    # With max aggregation:
    # A: max([1, 10, 3]) = 10
    # B: max([7, 2]) = 7
    # C: max([5]) = 5
    # So the order should be A, B, C
    assert labels_max.iloc[0]['label'] == 'A'
    assert labels_max.iloc[1]['label'] == 'B'
    assert labels_max.iloc[2]['label'] == 'C'

    # Test with min aggregation
    label_placer_min = LabelPlacement(
        data=df,
        by='labelA',
        x='x',
        y='y',
        importance='importance',
        importance_aggregation='min',
    )

    labels_min = label_placer_min.compute()

    # With min aggregation:
    # A: min([1, 10, 3]) = 1
    # B: min([7, 2]) = 2
    # C: min([5]) = 5
    # So the order should be C, B, A
    assert labels_min.iloc[0]['label'] == 'C'
    assert labels_min.iloc[1]['label'] == 'B'
    assert labels_min.iloc[2]['label'] == 'A'

    # Test with median aggregation
    label_placer_median = LabelPlacement(
        data=df,
        by='labelA',
        x='x',
        y='y',
        importance='importance',
        importance_aggregation='median',
    )

    labels_median = label_placer_median.compute()

    # With median aggregation:
    # A: median([1, 10, 3]) = 3
    # B: median([7, 2]) = 4.5
    # C: median([4]) = 4
    # So the order should be C, B, A
    assert labels_median.iloc[0]['label'] == 'B'
    assert labels_median.iloc[1]['label'] == 'C'
    assert labels_median.iloc[2]['label'] == 'A'

    # Test with custom aggregation function (standard deviation)
    label_placer_custom = LabelPlacement(
        data=df,
        by='labelA',
        x='x',
        y='y',
        importance='importance',
        importance_aggregation=lambda x: np.std(x),  # Standard deviation
    )

    labels_custom = label_placer_custom.compute()

    # With std aggregation:
    # A: std([1, 10, 3]) ≈ 4.73
    # B: std([7, 2]) ≈ 3.54
    # C: std([5]) = 0
    # So the order should be A, B, C
    assert labels_custom.iloc[0]['label'] == 'A'
    assert labels_custom.iloc[1]['label'] == 'B'
    assert labels_custom.iloc[2]['label'] == 'C'

    # Test with another custom function (range: max-min)
    label_placer_range = LabelPlacement(
        data=df,
        by='labelA',
        x='x',
        y='y',
        importance='importance',
        importance_aggregation=lambda x: np.max(x) - np.min(x),
    )

    labels_range = label_placer_range.compute()

    # With range aggregation:
    # A: max([1, 10, 3]) - min([1, 10, 3]) = 10 - 1 = 9
    # B: max([7, 2]) - min([7, 2]) = 7 - 2 = 5
    # C: max([5]) - min([5]) = 5 - 5 = 0
    # So the order should be A, B, C
    assert labels_range.iloc[0]['label'] == 'A'
    assert labels_range.iloc[1]['label'] == 'B'
    assert labels_range.iloc[2]['label'] == 'C'

    # Test with multiple label columns
    label_placer_multi = LabelPlacement(
        data=df,
        by=['labelA', 'labelB'],
        x='x',
        y='y',
        importance='importance',
        importance_aggregation='mean',
    )

    labels_multi = label_placer_multi.compute()

    # 'A2': mean([10]) = 10
    # 'B1': mean([7]) = 7
    # 'A': mean([1, 10, 3]) = 4.6666666667
    assert labels_multi.iloc[0]['label'] == 'A2'
    assert labels_multi.iloc[1]['label'] == 'B1'
    assert labels_multi.iloc[2]['label'] == 'A'

    # Test with multiple label columns but a different aggregation methid
    label_placer_multi_2 = label_placer_multi.clone(importance_aggregation='sum')
    labels_multi_2 = label_placer_multi_2.compute()

    # 'A': sum([1, 10, 3]) = 14
    # 'A2': sum([10]) = 10
    # 'B': sum([7, 2]) = 9
    assert labels_multi_2.iloc[0]['label'] == 'A'
    assert labels_multi_2.iloc[1]['label'] == 'A2'
    assert labels_multi_2.iloc[2]['label'] == 'B'
