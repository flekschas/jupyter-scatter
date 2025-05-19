def test_highest_density_point_functions():
    """Test highest density point calculation."""
    import numpy as np

    from jscatter.label_placement.highest_density_point import (
        compute_highest_density_point,
    )

    # Create a cluster of points
    np.random.seed(42)  # For reproducibility
    cluster1 = np.random.normal(loc=[0.2, 0.2], scale=0.05, size=(100, 2))
    cluster2 = np.random.normal(loc=[0.8, 0.8], scale=0.1, size=(20, 2))
    points = np.vstack([cluster1, cluster2])

    # Test highest desnity function
    result = compute_highest_density_point(points, grid_size=50)
    assert abs(result[0] - 0.2) < 0.1
    assert abs(result[1] - 0.2) < 0.1

    # Test with empty array
    empty_result = compute_highest_density_point(np.array([]))
    assert np.array_equal(empty_result, np.array([0, 0]))

    # Test with single point
    single_point = np.array([[1.0, 2.0]])
    single_result = compute_highest_density_point(single_point)
    assert np.array_equal(single_result, np.array([1.0, 2.0]))

    # Test with two points
    two_points = np.array([[1.0, 2.0], [3.0, 4.0]])
    two_result = compute_highest_density_point(two_points)
    assert np.array_equal(two_result, np.array([2.0, 3.0]))  # Should return mean


def test_highest_density_point_edge_cases():
    """Test edge cases and error handling in highest density point functions."""
    import numpy as np

    from jscatter.label_placement.highest_density_point import (
        compute_highest_density_point,
    )

    # Test with unusual grid size
    np.random.seed(42)
    cluster = np.random.normal(loc=[0.2, 0.2], scale=0.05, size=(5000, 2))
    result_small_grid = compute_highest_density_point(cluster, grid_size=5)
    result_large_grid = compute_highest_density_point(cluster, grid_size=200)

    # Both should find points near [0.2, 0.2] regardless of grid size
    assert abs(result_small_grid[0] - 0.2) < 0.15
    assert abs(result_small_grid[1] - 0.2) < 0.15
    assert abs(result_large_grid[0] - 0.2) < 0.05
    assert abs(result_large_grid[1] - 0.2) < 0.05
