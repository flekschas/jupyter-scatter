import math
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import brentq

from jscatter.label_placement.zoom_solver import (
    ASINH_1,
    MAX_SAFE_ZOOM,
    REGION_5_INVERSE,
    SMALLEST_SAFE_K,
    _f,
    _find_search_interval,
    _find_search_interval_with_grid,
    solve_zoom,
    solve_zoom_approximately,
    solve_zoom_precisely,
)


def test_zoom_solver_functions():
    """Test the zoom solver functions with various inputs."""

    # Test basic function behavior
    k_value = 0.5
    z_value = 2.0
    expected_f = math.asinh(z_value) / z_value - k_value

    assert np.isclose(_f(z_value, k_value), expected_f)

    # 1. Test k >= ASINH_1 cases (should return 1.0)
    assert solve_zoom(ASINH_1) == 1.0
    assert solve_zoom(0.9) == 1.0  # Since 0.9 > ASINH_1 ≈ 0.8814
    assert solve_zoom(0.99) == 1.0
    assert solve_zoom(1.0) == 1.0
    assert solve_zoom(10.0) == 1.0

    # The precise solver should also return 1.0 for these cases
    assert solve_zoom_precisely(ASINH_1) == 1.0
    assert solve_zoom_precisely(0.9) == 1.0
    assert solve_zoom_precisely(0.99) == 1.0
    assert solve_zoom_precisely(1.0) == 1.0

    # 2. Test k <= 0 cases (should raise ValueError)
    with pytest.raises(ValueError):
        solve_zoom(0.0)
    with pytest.raises(ValueError):
        solve_zoom(-0.1)

    # 3. Test precision for 0 < k < ASINH_1 with adaptive tolerance
    k_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.87, ASINH_1 - 0.01, ASINH_1 - 0.001]

    for k in k_values:
        # Verify k is actually in our valid range
        assert 0 < k < ASINH_1

        precise_result = solve_zoom_precisely(k)
        assert precise_result >= 1.0

        f_val = abs(_f(precise_result, k))

        # Define tolerance based on distance from ASINH_1
        distance_from_asinh1 = ASINH_1 - k

        if distance_from_asinh1 > 0.1:
            # Far from ASINH_1, should be high precision
            tolerance = 1e-9
        elif distance_from_asinh1 > 0.01:
            # Getting closer to ASINH_1
            tolerance = 0.02
        elif distance_from_asinh1 > 0.001:
            # Very close to ASINH_1
            tolerance = 0.05
        else:
            # Extremely close to ASINH_1
            tolerance = 0.1

        assert (
            f_val < tolerance
        ), f'Solution for k = {k} has error {f_val}, allowed tolerance {tolerance}'

    # 4. Test search interval functions
    interval = _find_search_interval(0.5)
    assert interval is not None
    assert len(interval) == 2
    assert interval[0] < interval[1]

    grid_interval = _find_search_interval_with_grid(0.5)
    assert grid_interval is not None
    assert len(grid_interval) == 2
    assert grid_interval[0] < grid_interval[1]

    # 5. Test brentq solver
    a, b = 1.0, 10.0
    result = brentq(lambda z: _f(z, 0.5), a, b, rtol=1e-10)
    assert a <= result <= b
    assert np.isclose(_f(result, 0.5), 0.0, atol=1e-8)

    # 6. Test approximate solver for valid range
    approx_k_values = [0.1, 0.3, 0.5, 0.7, 0.8, ASINH_1 - 0.01, ASINH_1 - 0.001]
    for k in approx_k_values:
        approx_result = solve_zoom_approximately(k)
        assert approx_result >= 1.0
        # Verify the result is reasonable
        f_val = abs(_f(approx_result, k))
        assert f_val < 0.2  # Approximate solution should be reasonably close


def test_zoom_solver_edge_cases():
    """Test edge cases and error handling in zoom solver functions."""

    # Test with k values that require backup strategies
    # Force brentq to fail, should fall back to approximation
    with patch(
        'scipy.optimize.brentq',
        side_effect=RuntimeError('Forced error'),
    ):
        result = solve_zoom_precisely(0.5)
        # Should still return a reasonable value
        assert result > 1.0
        assert abs(_f(result, 0.5)) < 0.2  # Approximate solution

    # Test search interval failures and fallbacks
    with patch(
        'jscatter.label_placement.zoom_solver._find_search_interval', return_value=None
    ):
        # Should fall back to grid search
        result = solve_zoom_precisely(0.5)
        assert result > 1.0

    # Test with k very close to 0 (but positive)
    very_small_k = 1e-6
    result = solve_zoom(very_small_k)
    assert result > 1000  # Should give a very large zoom level

    # Test the approximation regions explicitly
    # Region 1: ASINH_1 > k > 0.87
    k1 = 0.88
    r1 = solve_zoom_approximately(k1)
    assert r1 > 1.0 and r1 < 2.0

    # Region 2: 0.87 >= k > 0.7
    k2 = 0.75
    r2 = solve_zoom_approximately(k2)
    assert r2 > 1.0

    # Region 3: 0.7 >= k > 0.4
    k3 = 0.5
    r3 = solve_zoom_approximately(k3)
    assert r3 > 1.0

    # Region 4: 0.4 >= k > 0.1
    k4 = 0.2
    r4 = solve_zoom_approximately(k4)
    assert r4 > 1.0

    # Region 5: 0.1 >= k > 0
    k5 = 0.05
    r5 = solve_zoom_approximately(k5)
    assert r5 > 1.0

    # Test type checking for k
    with pytest.raises(TypeError):
        solve_zoom('not a number')

    # Test handling of extreme search intervals
    result = _find_search_interval_with_grid(
        0.0001, min_val=1.0, max_val=1000000.0, points=1000
    )
    assert result is not None
    assert len(result) == 2
    assert result[0] < result[1]

    # Test maximum iterations handling
    with patch(
        'jscatter.label_placement.zoom_solver._find_search_interval',
        return_value=None,
    ):
        with patch(
            'jscatter.label_placement.zoom_solver._find_search_interval_with_grid',
            return_value=None,
        ):
            # Should handle failure gracefully
            with pytest.raises(ValueError):
                solve_zoom_precisely(0.5)


def test_zoom_solver_exceptions():
    with pytest.raises(TypeError, match='Input k must be a number'):
        solve_zoom_approximately('Hallo')

    with pytest.raises(ValueError, match=r'Input k must be positive'):
        solve_zoom_approximately(0)

    assert solve_zoom_approximately(ASINH_1 + 1e-10) == 1

    assert solve_zoom_approximately(1e-12) == min(
        REGION_5_INVERSE / SMALLEST_SAFE_K, MAX_SAFE_ZOOM
    )

    with pytest.raises(ValueError, match=r'Input k must be positive'):
        solve_zoom_precisely(0)
