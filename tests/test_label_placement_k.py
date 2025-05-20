def test_compute_k_functions():
    """Test both JIT and non-JIT versions of compute_k function."""
    import numpy as np

    from jscatter.label_placement.k import (
        ASINH_1,
        compute_k,
    )

    # Basic test cases
    x1, w1 = 10.0, 5.0
    x2, w2 = 20.0, 3.0

    # Expected result
    expected = (abs(x1 - x2) * ASINH_1) / (w1 + w2)

    # Test k computation
    result = compute_k(x1, w1, x2, w2)
    assert np.isclose(result, expected)

    # Test with negative values
    neg_result = compute_k(-5.0, 2.0, 5.0, 3.0)
    neg_expected = (10.0 * ASINH_1) / 5.0
    assert np.isclose(neg_result, neg_expected)

    # Test with zero width
    zero_width_result = compute_k(10.0, 0.0, 20.0, 0.0)
    assert np.isinf(zero_width_result)  # Division by zero should give infinity


def test_compute_k_edge_cases():
    """Test edge cases and error handling in compute_k function."""

    import numpy as np

    from jscatter.label_placement.k import compute_k

    # Test with strange inputs that should still work
    result = compute_k(0.0, 1.0, 0.0, 1.0)  # Zero distance
    assert result == 0.0

    # Test with negative widths
    result = compute_k(10.0, -1.0, 20.0, -2.0)
    # Should use absolute values or treat as zero
    assert not np.isnan(result)

    # Test with very large values (check for overflow handling)
    result = compute_k(1e10, 1e5, 2e10, 1e5)
    assert not np.isnan(result) and not np.isinf(result)

    # Test with very small values (check for underflow handling)
    result = compute_k(1e-10, 1e-5, 2e-10, 1e-5)
    assert not np.isnan(result) and not np.isinf(result)
