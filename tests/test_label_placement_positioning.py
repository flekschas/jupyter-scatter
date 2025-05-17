import numpy as np
import pandas as pd

from jscatter.label_placement.label_placement import LabelPlacement


def test_label_positioning_options():
    """Test different label positioning options.

    We're going to great a grid with the following number and location of points:

     10 - 1                                                       1
      9 -
      8 -
      7 -
      6 -
      5 -
      4 -
      3 -
      2 -
      1 -
    0.9 -
    0.8 -
    0.7 -
    0.6 -
    0.5 - 1 1 1 1 1 1
    0.4 - 1 1 1 1 1 1
    0.3 - 1 1 1 1 1 1
    0.2 - 1 1 1 1 1 1
    0.1 - 1 1 1 1 1 1
      0 - 5 1 1 1 1 1                                             1
          |                   |   |   |   |   |   |   |   |   |   |
          0 . . . . . . . . . 1   2   3   4   5   6   7   8   9   10
    """

    points = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0.1],
            [0, 0.2],
            [0, 0.3],
            [0, 0.4],
            [0, 0.5],
            [0.1, 0],
            [0.1, 0.1],
            [0.1, 0.2],
            [0.1, 0.3],
            [0.1, 0.4],
            [0.1, 0.5],
            [0.2, 0],
            [0.2, 0.1],
            [0.2, 0.2],
            [0.2, 0.3],
            [0.2, 0.4],
            [0.2, 0.5],
            [0.3, 0],
            [0.3, 0.1],
            [0.3, 0.2],
            [0.3, 0.3],
            [0.3, 0.4],
            [0.3, 0.5],
            [0.4, 0],
            [0.4, 0.1],
            [0.4, 0.2],
            [0.4, 0.3],
            [0.4, 0.4],
            [0.4, 0.5],
            [0.5, 0],
            [0.5, 0.1],
            [0.5, 0.2],
            [0.5, 0.3],
            [0.5, 0.4],
            [0.5, 0.5],
            [10, 0],
            [10, 10],
            [0, 10],
        ]
    )
    category = ['test'] * len(points)
    df = pd.DataFrame(
        {
            'x': points[:, 0],
            'y': points[:, 1],
            'category': category,
        }
    )

    # Test center_of_mass positioning
    label_placer = LabelPlacement(
        data=df,
        by='category',
        x='x',
        y='y',
        tile_size=100,
        positioning='center_of_mass',
        bbox_percentile_range=(0, 100),
    )

    labels = label_placer._create_labels()

    assert len(labels) == 1, 'There should only be one label'
    label_position = labels.iloc[0][['x', 'y']].values

    # Verify center_of_mass position
    assert np.allclose(
        label_position, [5, 5], atol=1e-10
    ), f'Center of mass label position does not match [5, 5]'

    # Test highest_density positioning
    label_placer = LabelPlacement(
        data=df,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        positioning='highest_density',
        bbox_percentile_range=(0, 100),
    )

    # Run preprocessing to access the computed position
    labels = label_placer._create_labels()

    assert len(labels) == 1, 'There should only be one label'
    label_position = labels.iloc[0][['x', 'y']].values

    # The highest density should be in the first bin. Given that we use a 50x50
    # grid to determine the densities, the first grid goes from
    # [0, 10/50] = [0, 0.2]. The final position is the center of this bin, which
    # is [0.1, 0.1].
    assert np.allclose(
        label_position, [0.1, 0.1], atol=1e-10
    ), f'Highest density label position does not match [0.1, 0.1]'

    # Test largest_cluster positioning
    label_placer = LabelPlacement(
        data=df,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        positioning='largest_cluster',
        bbox_percentile_range=(0, 100),
    )

    # Run preprocessing to access the computed position
    labels = label_placer._create_labels()

    assert len(labels) == 1, 'There should only be one label'
    x, y = labels.iloc[0][['x', 'y']].values

    # The largest cluster center should also be somewhere in the middle of
    # [0, 0.5] as HDBSCAN should identify a cluster near zero. The exact points
    # that it consideres noise points are unimportant.
    assert x > 0 and x < 0.5, f'Largest cluster X label position is not in ]0, 0.5['
    assert y > 0 and y < 0.5, f'Largest cluster Y label position is not in ]0, 0.5['


def test_largest_cluster_function():
    """Test the compute_largest_cluster function with various inputs."""
    import numpy as np

    from jscatter.label_placement.largest_cluster import compute_largest_cluster

    # Create test points with multiple clusters
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[0.2, 0.2], scale=0.05, size=(50, 2))
    cluster2 = np.random.normal(loc=[0.8, 0.8], scale=0.05, size=(20, 2))
    outliers = np.array([[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])
    points = np.vstack([cluster1, cluster2, outliers])

    # Test with normal input
    result = compute_largest_cluster(points)
    assert len(result) > 0
    assert len(result) <= len(points)

    # Test with max_points parameter
    result_max_10 = compute_largest_cluster(points, max_points=10)
    assert len(result_max_10) <= 10

    # Test with empty array
    empty_result = compute_largest_cluster(np.array([]))
    assert np.array_equal(empty_result, np.array([0, 0]).reshape((1, 2)))

    # Test with small number of points
    two_points = np.array([[1.0, 2.0], [3.0, 4.0]])
    small_result = compute_largest_cluster(two_points)
    assert np.array_equal(small_result.reshape(-1), np.mean(two_points, axis=0))

    # Test with all noise points (where HDBSCAN would label all as noise)
    noise_points = np.array([[0, 0], [10, 10], [20, 20], [30, 30], [40, 40]])
    noise_result = compute_largest_cluster(noise_points)
    assert len(noise_result) >= 1  # Should return at least one point
