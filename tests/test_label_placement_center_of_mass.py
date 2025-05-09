from unittest.mock import patch

import numpy as np

from jscatter.label_placement.center_of_mass import (
    _compute_center_of_mass_jit,
    _compute_center_of_mass_legacy,
    compute_center_of_mass,
)


def test_center_of_mass_functions():
    """Test both JIT and non-JIT versions of center of mass calculation."""

    # Simple square test case
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # Test JIT version directly
    jit_result = _compute_center_of_mass_jit(square)
    assert np.allclose(jit_result, [0.5, 0.5])

    # Test legacy version
    legacy_result = _compute_center_of_mass_legacy(square)
    assert np.allclose(legacy_result, [0.5, 0.5])

    # Test wrapper function
    result = compute_center_of_mass(square)
    assert np.allclose(result, [0.5, 0.5])

    # Test with triangle
    triangle = np.array([[0, 0], [1, 0], [0.5, 1]])
    tri_result = compute_center_of_mass(triangle)
    assert np.allclose(tri_result, [0.5, 1 / 3])

    # Test with more complex polygon
    hexagon = np.array([[1, 0], [2, 0], [3, 1], [2, 2], [1, 2], [0, 1]])
    hex_result = compute_center_of_mass(hexagon)
    assert np.allclose(hex_result, [1.5, 1.0])

    # Test with edge case: zero area polygon (line)
    line = np.array([[0, 0], [1, 0], [2, 0]])
    line_result = compute_center_of_mass(line)
    assert np.allclose(line_result, [1.0, 0.0])


def test_center_of_mass_edge_cases():
    """Test edge cases and error handling in center of mass functions."""

    # Force JIT function to fail and ensure fallback works
    with patch(
        'jscatter.label_placement.center_of_mass._compute_center_of_mass_jit',
        side_effect=RuntimeError('Forced error'),
    ):
        result = compute_center_of_mass(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
        assert np.allclose(result, [0.5, 0.5])

    # Test with polygon where shoelace formula gives zero area
    collinear_points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    result = compute_center_of_mass(collinear_points)
    # Should fall back to mean when area is zero
    assert np.allclose(result, [1.5, 0])

    # Test with very small area polygon (near numerical instability)
    tiny_area = np.array([[0, 0], [1e-10, 0], [1e-10, 1e-10], [0, 1e-10]])
    result = compute_center_of_mass(tiny_area)
    # Should handle tiny area correctly without numerical issues
    assert result[0] >= 0 and result[0] <= 1e-10
    assert result[1] >= 0 and result[1] <= 1e-10
