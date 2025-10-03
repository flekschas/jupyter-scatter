"""
Tests for optional dependencies.

These tests verify that:
1. jscatter can be imported and basic functionality works without optional dependencies
2. Features requiring optional dependencies raise appropriate errors when dependencies are missing
"""

import sys
import pytest
import numpy as np
import pandas as pd

from jscatter.dependencies import DependencyError


def test_basic_import():
    """Test that jscatter can be imported without optional dependencies."""
    # This test will pass if we can import jscatter (which happens at the top of the file)
    import jscatter

    assert jscatter is not None


def test_basic_plot_without_optionals():
    """Test that basic plotting works without optional dependencies."""
    import jscatter

    # Test with simple arrays
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    scatter = jscatter.plot(x, y, show=False)
    assert scatter is not None

    # Test with DataFrame
    df = pd.DataFrame(
        {
            'x': np.random.rand(100),
            'y': np.random.rand(100),
            'value': np.random.rand(100),
        }
    )
    scatter = jscatter.plot(data=df, x='x', y='y', show=False)
    assert scatter is not None


@pytest.mark.skipif(
    'hdbscan' in sys.modules
    or __import__('importlib.util', fromlist=['find_spec']).find_spec('hdbscan')
    is not None,
    reason='hdbscan is installed',
)
def test_largest_cluster_requires_hdbscan():
    """Test that largest_cluster positioning raises error without hdbscan."""
    import jscatter

    df = pd.DataFrame(
        {'x': [0, 1, 1, 0], 'y': [0, 0, 1, 1], 'label': ['A', 'A', 'B', 'B']}
    )

    with pytest.raises(DependencyError) as exc_info:
        scatter = jscatter.plot(
            data=df,
            x='x',
            y='y',
            label_by='label',
            label_positioning='largest_cluster',
            show=False,
        )

    assert 'hdbscan' in str(exc_info.value).lower()
    assert 'label-extras' in str(exc_info.value) or 'all' in str(exc_info.value)


@pytest.mark.skipif(
    'seaborn' in sys.modules
    or __import__('importlib.util', fromlist=['find_spec']).find_spec('seaborn')
    is not None,
    reason='seaborn is installed',
)
def test_contour_requires_seaborn():
    """Test that contour annotations require seaborn."""
    import jscatter
    from jscatter.composite_annotations import Contour

    df = pd.DataFrame(
        {
            'x': np.random.rand(100),
            'y': np.random.rand(100),
        }
    )

    with pytest.raises(DependencyError) as exc_info:
        contour = Contour(df['x'], df['y'])

    assert 'seaborn' in str(exc_info.value).lower()
    assert 'annotation-extras' in str(exc_info.value) or 'all' in str(exc_info.value)


def test_label_placement_without_progress_bar():
    """Test that label placement works without tqdm (no progress bar)."""
    import jscatter

    df = pd.DataFrame(
        {
            'x': [0, 1, 2, 0, 1, 2],
            'y': [0, 0, 0, 1, 1, 1],
            'label': ['A', 'B', 'C', 'D', 'E', 'F'],
        }
    )

    # This should work without tqdm (just no progress bar)
    scatter = jscatter.plot(data=df, x='x', y='y', label_by='label', show=False)

    # Computing label placement without progress bar should work
    if hasattr(scatter, 'label_placement') and scatter.label_placement is not None:
        # This should not raise an error, even without tqdm
        # (it just won't show a progress bar)
        scatter.label_placement.compute(show_progress=False)
