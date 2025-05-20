import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from test_label_placement import sample_data

from jscatter.label_placement.label_placement import LabelPlacement


@pytest.fixture
def label_placement():
    """Create a simple label placement instance for testing label persistence."""
    df = pd.DataFrame(
        {
            'x': [0.1, 0.2, 0.3, 0.4, 0.5],
            'y': [0.1, 0.2, 0.3, 0.4, 0.5],
            'category': ['A', 'B', 'C', 'D', 'E'],
            'importance': [5, 4, 3, 2, 1],
        }
    )

    # Create a label placer
    label_placer = LabelPlacement(
        data=df,
        by='category',
        x='x',
        y='y',
        tile_size=256,
        importance='importance',
        scale_function='constant',
    )

    # Compute labels
    label_placer.compute()

    return label_placer


def test_to_parquet_from_parquet(label_placement: LabelPlacement):
    """Test saving to and loading from parquet format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save to parquet
        label_placement.to_parquet(temp_dir, format='parquet')

        # Verify files exist
        assert os.path.exists(os.path.join(temp_dir, 'label-data.parquet'))
        assert os.path.exists(os.path.join(temp_dir, 'label-tiles.parquet'))

        # Load from parquet
        loaded_label_placer = LabelPlacement.from_parquet(temp_dir, format='parquet')

        # Verify the loaded data matches the original
        assert loaded_label_placer.tile_size == label_placement.tile_size
        assert loaded_label_placer._x_min == label_placement._x_min
        assert loaded_label_placer._x_max == label_placement._x_max
        assert loaded_label_placer._y_min == label_placement._y_min
        assert loaded_label_placer._y_max == label_placement._y_max
        assert loaded_label_placer._x_extent == label_placement._x_extent
        assert loaded_label_placer._y_extent == label_placement._y_extent
        assert loaded_label_placer.positioning == label_placement.positioning
        assert loaded_label_placer.scale_function == label_placement.scale_function
        assert (
            loaded_label_placer.max_labels_per_tile
            == label_placement.max_labels_per_tile
        )

        assert loaded_label_placer.labels is not None
        assert label_placement.labels is not None

        # Check that the label data is the same
        pd.testing.assert_frame_equal(
            loaded_label_placer.labels.reset_index(drop=True),
            label_placement.labels.reset_index(drop=True),
        )

        assert loaded_label_placer.tiles is not None
        assert label_placement.tiles is not None

        # Check that the tiles are the same
        assert set(loaded_label_placer.tiles.keys()) == set(
            label_placement.tiles.keys()
        )
        for tile_id in label_placement.tiles:
            assert set(loaded_label_placer.tiles[tile_id]) == set(
                label_placement.tiles[tile_id]
            )


def test_to_arrow_ipc_from_arrow_ipc(label_placement: LabelPlacement):
    """Test saving to and loading from Arrow IPC format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save to Arrow IPC
        label_placement.to_parquet(temp_dir, format='arrow_ipc')

        # Verify files exists
        assert os.path.exists(os.path.join(temp_dir, 'label-data.arrow'))
        assert os.path.exists(os.path.join(temp_dir, 'label-tiles.arrow'))

        # Load from Arrow IPC
        loaded_label_placer = LabelPlacement.from_parquet(temp_dir, format='arrow_ipc')

        # Verify the loaded data matches the original
        assert loaded_label_placer.tile_size == label_placement.tile_size
        assert loaded_label_placer._x_min == label_placement._x_min
        assert loaded_label_placer._x_max == label_placement._x_max
        assert loaded_label_placer._y_min == label_placement._y_min
        assert loaded_label_placer._y_max == label_placement._y_max
        assert loaded_label_placer._x_extent == label_placement._x_extent
        assert loaded_label_placer._y_extent == label_placement._y_extent
        assert loaded_label_placer.positioning == label_placement.positioning
        assert loaded_label_placer.scale_function == label_placement.scale_function
        assert (
            loaded_label_placer.max_labels_per_tile
            == label_placement.max_labels_per_tile
        )
        assert loaded_label_placer.importance == label_placement.importance
        assert (
            loaded_label_placer.importance_aggregation
            == label_placement.importance_aggregation
        )
        assert (
            loaded_label_placer.target_aspect_ratio
            == label_placement.target_aspect_ratio
        )
        assert loaded_label_placer.max_lines == label_placement.max_lines
        assert loaded_label_placer.background == label_placement.background
        assert loaded_label_placer.size == label_placement.size
        assert loaded_label_placer.color == label_placement.color
        assert loaded_label_placer.zoom_range == label_placement.zoom_range
        assert loaded_label_placer.exclude == label_placement.exclude
        assert loaded_label_placer.font == label_placement.font

        assert label_placement.labels is not None
        assert loaded_label_placer.labels is not None

        # Check that the label data is the same
        pd.testing.assert_frame_equal(
            loaded_label_placer.labels.reset_index(drop=True),
            label_placement.labels.reset_index(drop=True),
        )

        assert label_placement.tiles is not None
        assert loaded_label_placer.tiles is not None

        # Check that the tiles are the same
        assert set(loaded_label_placer.tiles.keys()) == set(
            label_placement.tiles.keys()
        )
        for tile_id in label_placement.tiles:
            assert set(loaded_label_placer.tiles[tile_id]) == set(
                label_placement.tiles[tile_id]
            )


def test_with_path_prefix(label_placement: LabelPlacement):
    """Test saving to and loading from parquet format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Path prefix
        path_prefix = temp_dir + '/custom'

        # Save to parquet
        label_placement.to_parquet(path_prefix, format='parquet')

        # Verify files exist
        assert os.path.exists(path_prefix + '-label-data.parquet')
        assert os.path.exists(path_prefix + '-label-tiles.parquet')

        # Load from parquet
        loaded_label_placer = LabelPlacement.from_parquet(path_prefix, format='parquet')

        # Verify the loaded data matches the original
        assert loaded_label_placer.tile_size == label_placement.tile_size
        assert loaded_label_placer._x_min == label_placement._x_min
        assert loaded_label_placer._x_max == label_placement._x_max
        assert loaded_label_placer._y_min == label_placement._y_min
        assert loaded_label_placer._y_max == label_placement._y_max
        assert loaded_label_placer._x_extent == label_placement._x_extent
        assert loaded_label_placer._y_extent == label_placement._y_extent
        assert loaded_label_placer.positioning == label_placement.positioning
        assert loaded_label_placer.scale_function == label_placement.scale_function
        assert (
            loaded_label_placer.max_labels_per_tile
            == label_placement.max_labels_per_tile
        )
        assert loaded_label_placer.importance == label_placement.importance
        assert (
            loaded_label_placer.importance_aggregation
            == label_placement.importance_aggregation
        )
        assert (
            loaded_label_placer.target_aspect_ratio
            == label_placement.target_aspect_ratio
        )
        assert loaded_label_placer.max_lines == label_placement.max_lines
        assert loaded_label_placer.background == label_placement.background
        assert loaded_label_placer.size == label_placement.size
        assert loaded_label_placer.color == label_placement.color
        assert loaded_label_placer.zoom_range == label_placement.zoom_range
        assert loaded_label_placer.exclude == label_placement.exclude
        assert loaded_label_placer.font == label_placement.font

        assert loaded_label_placer.labels is not None
        assert label_placement.labels is not None

        # Check that the label data is the same
        pd.testing.assert_frame_equal(
            loaded_label_placer.labels.reset_index(drop=True),
            label_placement.labels.reset_index(drop=True),
        )

        assert loaded_label_placer.tiles is not None
        assert label_placement.tiles is not None

        # Check that the tiles are the same
        assert set(loaded_label_placer.tiles.keys()) == set(
            label_placement.tiles.keys()
        )
        for tile_id in label_placement.tiles:
            assert set(loaded_label_placer.tiles[tile_id]) == set(
                label_placement.tiles[tile_id]
            )


def test_persistence_auto_format_detection(label_placement: LabelPlacement):
    """Test automatic format detection when loading."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save as Parquet format
        parquet_dir = os.path.join(temp_dir, 'parquet_data')
        label_placement.to_parquet(parquet_dir, format='parquet')

        # Save as Arrow IPC format
        arrow_dir = os.path.join(temp_dir, 'arrow_data')
        label_placement.to_parquet(arrow_dir, format='arrow_ipc')

        # Load with automatic format detection
        loaded_from_parquet = LabelPlacement.from_parquet(
            parquet_dir
        )  # Should detect as Parquet
        loaded_from_arrow = LabelPlacement.from_parquet(
            arrow_dir
        )  # Should detect as Arrow IPC

        # Verify both loaded correctly
        assert loaded_from_parquet.tile_size == label_placement.tile_size
        assert loaded_from_arrow.tile_size == label_placement.tile_size

        assert label_placement.labels is not None
        assert loaded_from_parquet.labels is not None
        assert loaded_from_arrow.labels is not None

        # Check that the label data is the same for both
        pd.testing.assert_frame_equal(
            loaded_from_parquet.labels.reset_index(drop=True),
            label_placement.labels.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            loaded_from_arrow.labels.reset_index(drop=True),
            label_placement.labels.reset_index(drop=True),
        )


def test_persistence_errors():
    """Test error handling for persistence functions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test loading with invalid format
        with pytest.raises(ValueError):
            LabelPlacement.from_parquet(temp_dir, format='invalid_format')

        # Test loading non-existent file
        non_existent_path = os.path.join(temp_dir, 'does_not_exist')
        with pytest.raises(FileNotFoundError):
            LabelPlacement.from_parquet(non_existent_path)

        # Create an empty directory to test loading with missing metadata
        empty_dir = os.path.join(temp_dir, 'empty_dir')
        os.makedirs(empty_dir)
        with pytest.raises(FileNotFoundError):
            LabelPlacement.from_parquet(empty_dir)


def test_persistence_with_different_tile_sizes():
    """Test persistence with different tile sizes."""
    df = pd.DataFrame(
        {
            'x': [0.1, 0.2, 0.3, 0.4, 0.5],
            'y': [0.1, 0.2, 0.3, 0.4, 0.5],
            'category': ['A', 'B', 'C', 'D', 'E'],
            'importance': [5, 4, 3, 2, 1],
        }
    )

    # Create a label placer with a large tile size
    large_tile_placer = LabelPlacement(
        data=df,
        by='category',
        x='x',
        y='y',
        tile_size=1024,
        importance='importance',
    )
    large_tile_placer.compute()

    # Create a label placer with a small tile size
    small_tile_placer = LabelPlacement(
        data=df,
        by='category',
        x='x',
        y='y',
        tile_size=64,
        importance='importance',
    )
    small_tile_placer.compute()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save both to parquet
        large_tile_path = os.path.join(temp_dir, 'large_tile')
        small_tile_path = os.path.join(temp_dir, 'small_tile')

        large_tile_placer.to_parquet(large_tile_path)
        small_tile_placer.to_parquet(small_tile_path)

        # Load both
        loaded_large = LabelPlacement.from_parquet(large_tile_path)
        loaded_small = LabelPlacement.from_parquet(small_tile_path)

        # Verify the tile sizes were preserved
        assert loaded_large.tile_size == 1024
        assert loaded_small.tile_size == 64

        assert loaded_large.labels is not None
        assert loaded_small.labels is not None

        # The zoom levels should be different due to different tile sizes
        assert not loaded_large.labels['zoom_in'].equals(loaded_small.labels['zoom_in'])


def test_complex_data_persistence(sample_data):
    """Test persistence with more complex data from the fixture."""
    # Use the existing sample_data fixture from other tests
    label_placer = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        tile_size=256,
        hierarchical=True,
    )

    label_placer.compute()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test both formats
        parquet_dir = os.path.join(temp_dir, 'parquet_data')
        arrow_dir = os.path.join(temp_dir, 'arrow_data')

        # Save both formats
        label_placer.to_parquet(parquet_dir, format='parquet')
        label_placer.to_parquet(arrow_dir, format='arrow_ipc')

        # Load both formats
        loaded_parquet = LabelPlacement.from_parquet(parquet_dir)
        loaded_arrow = LabelPlacement.from_parquet(arrow_dir)

        # Check hierarchical parameter
        assert loaded_parquet.hierarchical == label_placer.hierarchical
        assert loaded_arrow.hierarchical == label_placer.hierarchical

        # Check by columns
        assert loaded_parquet.by == label_placer.by
        assert loaded_arrow.by == label_placer.by

        assert loaded_parquet.labels is not None
        assert loaded_arrow.labels is not None
        assert label_placer.labels is not None

        # Check that the label data has the correct structure
        assert set(loaded_parquet.labels['label_type'].unique()) == set(
            label_placer.labels['label_type'].unique()
        )
        assert set(loaded_arrow.labels['label_type'].unique()) == set(
            label_placer.labels['label_type'].unique()
        )


def test_persistence_with_custom_importance_aggregation():
    """Test persistence with a custom importance aggregation function."""

    # Custom aggregation function
    def custom_aggregation(values):
        # Something unique that can't be represented by standard aggregations
        return np.percentile(values, 75)

    df = pd.DataFrame(
        {
            'x': [0.1, 0.2, 0.3, 0.4, 0.5],
            'y': [0.1, 0.2, 0.3, 0.4, 0.5],
            'category': ['A', 'B', 'C', 'D', 'E'],
            'importance': [5, 4, 3, 2, 1],
        }
    )

    # Create a label placer with a custom importance aggregation
    label_placer = LabelPlacement(
        data=df,
        by='category',
        x='x',
        y='y',
        importance='importance',
        importance_aggregation=custom_aggregation,  # Custom function
    )

    # Compute labels
    label_placer.compute()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Capture warnings to check if the custom function warning is issued
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            # Save to parquet
            label_placer.to_parquet(temp_dir, format='parquet')

            # Check if warning was issued
            assert any(
                'Custom `importance_aggregation` function' in str(warning.message)
                for warning in w
            )

        # Load from parquet
        loaded_label_placer = LabelPlacement.from_parquet(temp_dir, format='parquet')

        # Verify that the default importance_aggregation was used
        assert loaded_label_placer.importance_aggregation == 'mean'

        # Other properties should still match
        assert loaded_label_placer.importance == label_placer.importance


def test_persistence_with_custom_font():
    """Test persistence with a custom font that's not in the font_name_map."""
    from jscatter.font import Font

    # Create a custom font that's not in the font_name_map
    custom_font = Font(
        spec_file_path='arial.json',
        face='CustomFontName',
        style='normal',
        weight='normal',
    )

    df = pd.DataFrame(
        {
            'x': [0.1, 0.2, 0.3, 0.4, 0.5],
            'y': [0.1, 0.2, 0.3, 0.4, 0.5],
            'category': ['A', 'B', 'C', 'D', 'E'],
        }
    )

    # Create a label placer with a custom font
    label_placer = LabelPlacement(
        data=df,
        by='category',
        x='x',
        y='y',
        font=custom_font,
    )

    # Compute labels
    label_placer.compute()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Capture warnings to check if the custom font warning is issued
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            # Save to parquet
            label_placer.to_parquet(temp_dir, format='parquet')

            # Check if warning was issued
            assert any(
                'Custom fonts cannot be serialized' in str(warning.message)
                for warning in w
            )

        # Load from parquet
        loaded_label_placer = LabelPlacement.from_parquet(temp_dir, format='parquet')

        # Verify that the fallback font (arial) was used
        from jscatter.font import font_name_map

        for key in loaded_label_placer.font:
            assert loaded_label_placer.font[key].face == font_name_map['arial'].face
