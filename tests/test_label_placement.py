import math
import re
from typing import cast

import numpy as np
import pandas as pd
import pytest

from jscatter.font import arial
from jscatter.label_placement.label_placement import LabelPlacement


@pytest.fixture
def sample_data():
    """Create sample data for testing label placement."""
    # Create a dataframe with coordinates and categorical columns
    np.random.seed(42)  # For reproducibility
    n = 100

    categories = ['A', 'B', 'C', 'D']
    subcategories = ['1', '2', '3']

    df = pd.DataFrame(
        {
            'x': np.random.uniform(0, 1, n),
            'y': np.random.uniform(0, 1, n),
            'category': np.random.choice(categories, n),
            'subcategory': [''] * n,
            'importance': np.random.uniform(0, 1, n),
        }
    )

    # Ensure subcategories are properly nested within categories
    # Each subcategory should belong to only one category
    for category in categories:
        df_cat = df['category'] == category
        subcategories = [f'{category}{s}' for s in ['1', '2', '3']]
        df.loc[df_cat, 'subcategory'] = np.random.choice(subcategories, len(df[df_cat]))

    df.index.name = 'id'
    df = df.reset_index()

    return df


def test_initialization(sample_data):
    """Test that LabelPlacement initializes correctly with default parameters."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
    )

    assert isinstance(label_placer, LabelPlacement)
    assert label_placer.data.equals(sample_data)
    assert label_placer.x == 'x'
    assert label_placer.y == 'y'
    assert label_placer.by == ['category']
    assert label_placer.tile_size == 256


def test_initialization_with_multiple_hierarchy_levels(sample_data):
    """Test initialization with multiple hierarchy levels."""
    label_placer = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        tile_size=600,
        hierarchical=True,
    )

    assert label_placer.by == ['category', 'subcategory']

    label_placer = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory', 'id'],
        x='x',
        y='y',
        tile_size=600,
        hierarchical=True,
    )

    assert label_placer.by == ['category', 'subcategory', 'id']


def test_initialization_with_importance(sample_data):
    """Test initialization with importance column."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        importance='importance',
    )

    assert label_placer.importance == 'importance'


def test_initialization_with_custom_fonts(sample_data):
    """Test initialization with custom font."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        font=arial.bold,
    )

    assert label_placer.font['category'] == arial.bold

    label_placer = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        tile_size=600,
        font=[arial.bold, arial.regular],
    )

    assert label_placer.font == {
        'category': arial.bold,
        'subcategory': arial.regular,
    }


def test_initialization_with_custom_sizes(sample_data):
    """Test initialization with custom font size."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        size=20,
    )

    assert label_placer.size['category'] == 20

    label_placer = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        tile_size=600,
        size=[24, 16],
    )

    assert label_placer.size == {'category': 24, 'subcategory': 16}


def test_initialization_with_custom_colors(sample_data):
    """Test initialization with custom color."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        color='red',
    )

    # Check that all labels have the same color
    assert all(color == '#ff0000' for color in label_placer.color.values())

    label_placer = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        tile_size=600,
        color=['red', (0.0, 1.0, 0.0)],
    )

    # Check that all labels are either red or green
    assert all(
        color == '#ff0000' or color == '#00ff00'
        for color in label_placer.color.values()
    )


def test_initialization_with_color_dict(sample_data):
    """Test initialization with color dictionary."""
    color_map = {
        'A': 'red',  # named color
        'B': (0.0, 1.0, 0.0),  # RGB color
        'C': '#DA00DB',  # HEX color
        'D': (254 / 255, 193 / 255, 14 / 255, 0.5),  # RGBA color
    }

    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        color=color_map,
    )

    # Check that the default color is black (as the default background is white)
    assert label_placer.color['category'] == '#000000'  # black
    # Check colors for specific categories
    assert label_placer.color['category:A'] == '#ff0000'  # red
    assert label_placer.color['category:B'] == '#00ff00'  # green
    assert label_placer.color['category:C'] == '#da00db'  # cyan/pink
    assert label_placer.color['category:D'] == '#fec10e'  # yellow

    color_map['category'] = 'gray'

    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        color=color_map,
    )

    # Check that the default color is gray now
    assert label_placer.color['category'] == '#808080'  # gray


def test_validate_data_with_missing_columns(sample_data):
    """Test that validation fails with missing columns."""
    # Create data missing the y column
    bad_data = sample_data.drop(columns=['y'])

    with pytest.raises(ValueError, match='Missing columns in data'):
        LabelPlacement(
            data=bad_data,
            by='category',
            x='x',
            y='y',  # This column is missing
            tile_size=600,
        )


def test_validate_data_with_hierarchy_violation(sample_data):
    """Test that validation fails with hierarchy violations."""
    # Create data with hierarchy violation (same subcategory in multiple categories)
    bad_data = sample_data.copy()
    bad_data.loc[bad_data['category'] == 'A', 'subcategory'] = 'Z'
    # Violation: Z belongs to both A and C
    bad_data.loc[bad_data['category'] == 'C', 'subcategory'] = 'Z'

    # Without proclaiming hierarchical data, the label placer should be okay
    LabelPlacement(
        data=bad_data,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        tile_size=600,
    )

    with pytest.raises(ValueError, match='Hierarchy violation'):
        LabelPlacement(
            data=bad_data,
            by=['category', 'subcategory'],
            hierarchical=True,
            x='x',
            y='y',
            tile_size=600,
        )


def test_compute_bbox(sample_data):
    """Test computation of bounding boxes and center of mass."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        positioning='center_of_mass',
    )

    # Test with simple points
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    bbox, center_mass = label_placer._compute_bbox(points)

    # Check bounding box
    assert bbox[0] == 0  # min_x
    assert bbox[1] == 0  # min_y
    assert bbox[2] == 1  # max_x
    assert bbox[3] == 1  # max_y

    # Check center of mass (should be [0.5, 0.5] for a square)
    assert np.isclose(center_mass[0], 0.5)
    assert np.isclose(center_mass[1], 0.5)

    # Test with empty points
    empty_points = np.array([])
    bbox, center_mass = label_placer._compute_bbox(empty_points)
    assert bbox[0] == 0  # min_x
    assert bbox[1] == 0  # min_y
    assert bbox[2] == 0  # max_x
    assert bbox[3] == 0  # max_y
    assert center_mass[0] == 0
    assert center_mass[1] == 0

    # Test with single point
    single_point = np.array([[0.5, 0.5]])
    bbox, center_mass = label_placer._compute_bbox(single_point)
    assert bbox[0] == 0.5  # min_x
    assert bbox[1] == 0.5  # min_y
    assert bbox[2] == 0.5  # max_x
    assert bbox[3] == 0.5  # max_y
    assert center_mass[0] == 0.5
    assert center_mass[1] == 0.5


def test_create_labels(sample_data):
    """Test label preprocessing."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
    )

    # Run preprocessing
    labels = label_placer._create_labels()

    # Basic validation
    assert isinstance(labels, pd.DataFrame)
    assert not labels.empty
    assert set(labels.columns).issuperset(
        {
            'label',
            'label_type',
            'importance',
            'hash',
            'x',
            'y',
            'bbox_width',
            'bbox_height',
            'font_color',
            'font_face',
            'font_style',
            'font_weight',
            'font_size',
        }
    )

    # Check number of labels (should have one per unique category)
    assert len(labels) == len(sample_data['category'].unique())

    # Check that labels are sorted descendingly by importance
    assert labels['importance'].is_monotonic_decreasing

    label_placer = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        hierarchical=True,
        x='x',
        y='y',
        tile_size=600,
    )
    labels = label_placer._create_labels()

    # Even for multiple label types, labels should be sorted descendingly by importance
    assert labels['importance'].is_monotonic_decreasing

    label_placer = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        hierarchical=True,
        x='x',
        y='y',
        tile_size=600,
    )
    labels = label_placer._create_labels()

    # However, for hierarchical labels, check that labels are sorted ascendingly by hierarchy_level
    assert labels['label_type'].is_monotonic_increasing


def test_compute_zoom_levels(sample_data):
    """Test computation of zoom levels."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
    )

    # Run preprocessing to get label data
    labels = label_placer._create_labels()

    # Compute zoom levels
    labels = label_placer._compute_zoom_levels(labels)

    # Check that zoom levels were computed
    assert 'width' in labels.columns
    assert 'height' in labels.columns
    assert 'zoom_in' in labels.columns
    assert 'zoom_out' in labels.columns

    # Basic validation of zoom values
    assert all(labels['zoom_in'] > 0)
    assert all(labels['zoom_out'] >= labels['zoom_in'])


def test_process_empty_data():
    """Test processing with empty data."""
    empty_df = pd.DataFrame(columns=['x', 'y', 'category'])

    label_placer = LabelPlacement(
        data=empty_df,
        by='category',
        x='x',
        y='y',
        tile_size=600,
    )

    result = label_placer.compute()

    # Should return an empty labels
    assert result.empty


def test_process(sample_data):
    """Test the full processing pipeline."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
    )

    result = label_placer.compute()

    # Check that result contains expected data
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert all(
        col in result.columns
        for col in [
            'label',
            'label_type',
            'importance',
            'hash',
            'x',
            'y',
            'bbox_width',
            'bbox_height',
            'font_color',
            'font_face',
            'font_style',
            'font_weight',
            'font_size',
            'width',
            'height',
            'zoom_in',
            'zoom_out',
            'zoom_fade_extent',
        ]
    )

    # Check that zoom levels are valid
    assert all(result['zoom_in'] > 0 | np.isinf(result['zoom_in']))
    assert all(result['zoom_out'] >= 0)


def test_process_overlapping_labels(sample_data):
    """Test the hiding of perfectly overlapping labels."""
    bad_data = sample_data.copy()
    # Duplicate last row
    bad_data = pd.concat([bad_data, bad_data.iloc[[-1]]], ignore_index=True)
    bad_data.loc[bad_data.iloc[[-1]].index, 'id'] += 1
    bad_data.iloc[[-2]].id

    label_placer = LabelPlacement(
        data=bad_data,
        by='id',
        x='x',
        y='y',
        importance='id',
        tile_size=600,
    )

    labels = label_placer.compute()

    second_last_id = int(bad_data.iloc[-2].id)
    second_last_label = labels.query(f'label == "{second_last_id}"').iloc[0]

    # Check that the last row's zoom in and out levels are set to infinity.
    # I.e. the label is hidden
    assert math.isinf(second_last_label['zoom_in'])
    assert math.isinf(second_last_label['zoom_out'])


def test_find_data_tile(sample_data):
    """Test finding data in specific tiles, including edge cases for tile lookup."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
    )

    # Process data to generate tiles
    label_placer.compute()

    # Verify computed state
    assert label_placer.computed == True
    assert label_placer.labels is not None
    assert label_placer.tiles is not None
    assert len(label_placer.tiles) > 0

    # Case 1: Test finding a tile that exists directly
    first_tile_id = next(iter(label_placer.tiles.keys()))
    result = label_placer._find_data_tile(first_tile_id)
    assert isinstance(result, list)
    assert len(result) > 0

    # Parse a sample tile to understand format
    x, y, z = map(int, first_tile_id.split(','))

    # Case 2: Test tile with negative coordinates
    negative_tile_id = f'-1,{y},{z}'
    result = label_placer._find_data_tile(negative_tile_id)
    assert result is None, 'Negative x coordinate should return None'

    # Case 3: Test tile with out-of-bounds coordinates
    max_xy = 2**z  # This is what zoom_level_to_zoom_scale(z) returns
    out_of_bounds_tile_id = f'{max_xy + 1},{y},{z}'
    result = label_placer._find_data_tile(out_of_bounds_tile_id)
    assert result is None, 'X coordinate > max_xy should return None'

    out_of_bounds_tile_id = f'{x},{max_xy + 1},{z}'
    result = label_placer._find_data_tile(out_of_bounds_tile_id)
    assert result is None, 'Y coordinate > max_xy should return None'

    # Case 4: Test finding parent tile
    # Create a child tile ID that doesn't exist but whose parent does
    parent_tile_found = False

    highest_z_tile_id = [0, 0, 0]

    # Find a tile at zoom level > 0
    for tile_id_str in label_placer.tiles.keys():
        tile_id = list(map(int, tile_id_str.split(',')))
        if tile_id[2] > highest_z_tile_id[2]:
            highest_z_tile_id = tile_id

    highest_z_tile_id_str = ','.join(map(str, highest_z_tile_id))
    highest_z_result = label_placer._find_data_tile(highest_z_tile_id_str)

    # Create a child tile that doesn't exist
    child_tile_id = [
        highest_z_tile_id[0] * 2,
        highest_z_tile_id[1] * 2,
        highest_z_tile_id[2] + 1,
    ]
    child_tile_id_str = ','.join(map(str, child_tile_id))

    # Double check that this tile is not in `tiles`
    assert child_tile_id_str not in label_placer.tiles

    # Try to find data for child tile
    child_result = label_placer._find_data_tile(child_tile_id_str)

    # Should have found the direct parent
    assert highest_z_result == child_result

    # Case 5: Test finding a tile that requires multiple parent lookups
    # Try to find a tile that's multiple levels above any existing tile
    grand_child_tile_id = [
        highest_z_tile_id[0] * 4,
        highest_z_tile_id[1] * 4,
        highest_z_tile_id[2] + 2,
    ]
    grand_child_tile_id_str = ','.join(map(str, grand_child_tile_id))

    # Try to find data for grand child tile
    grand_child_result = label_placer._find_data_tile(grand_child_tile_id_str)

    # Should have found the grand parent tile
    assert highest_z_result == grand_child_result

    # Case 6: Test finding a tile that doesn't exist at any level
    # Create a tile ID for a completely different region
    diff_region_x = int(1e6)  # Some very large value
    diff_region_y = int(1e6)
    diff_region_z = 10
    diff_region_tile_id = f'{diff_region_x},{diff_region_y},{diff_region_z}'

    # Should return None as this region doesn't exist at any zoom level
    result = label_placer._find_data_tile(diff_region_tile_id)
    assert result is None, 'Should return None for completely different region'

    # Case 7: Test when z becomes 0 and tile still doesn't exist
    # This should happen when we've walked all the way up the hierarchy
    # Create a tile at z=1 that won't have a parent in the tiles
    missing_base_tile_id = '1000,1000,1'
    result = label_placer._find_data_tile(missing_base_tile_id)
    assert result is None, 'Should return None when reaching z=0 without finding tile'

    # Case 8: Test behavior with an incorrectly formatted tile_id
    with pytest.raises(ValueError):
        label_placer._find_data_tile('invalid_format')


def test_get_labels_from_tiles(sample_data):
    """Test getting labels from specific tiles."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
    )

    # Process data to generate tiles
    label_placer.compute()
    assert label_placer.tiles is not None

    # Get all tile IDs
    tile_ids = list(label_placer.tiles.keys())

    if tile_ids:
        # Get labels from the first tile
        result = label_placer.get_labels_from_tiles([tile_ids[0]])

        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)


def test_performance_with_larger_dataset():
    """Test performance with a larger dataset."""
    # Create a larger dataset
    np.random.seed(42)
    n = 1000

    large_df = pd.DataFrame(
        {
            'x': np.random.uniform(0, 1, n),
            'y': np.random.uniform(0, 1, n),
            'category': np.random.choice(
                ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], n
            ),
        }
    )

    label_placer = LabelPlacement(
        data=large_df,
        by='category',
        x='x',
        y='y',
        tile_size=600,
    )

    import time

    start_time = time.time()
    label_placer.compute()
    end_time = time.time()

    # Just check that it completes in a reasonable time
    processing_time = end_time - start_time
    assert processing_time < 1.0, f'Processing took too long: {processing_time} seconds'


def test_label_collision_resolution():
    """Test collision resolutions."""
    np.random.seed(42)

    df = pd.DataFrame(
        {
            'x': [0, 10, 10, 0, 0],
            'y': [0, 0, 10, 10, 0.1],
            'label': ['test01', 'test23', 'test45', 'test67', 'test89'],
            'importance': [5, 4, 3, 2, 1],
        }
    )

    label_placer = LabelPlacement(
        data=df,
        x='x',
        y='y',
        by='label',
        importance='importance',
        scale_function='constant',
        tile_size=100,
    )

    labels = label_placer.compute()

    h0, zoom_in1_first = labels.iloc[0][['height', 'zoom_in']]
    h4, zoom_in4_first = labels.iloc[4][['height', 'zoom_in']]

    # Since we're using constant zoom scaling and the x position of the first
    # and fifth label perfectly overlap, the zoom level at which the collision
    # is resolved can be calculated as follows
    resolution_zoom = h0 / 0.1

    assert zoom_in1_first < zoom_in4_first
    assert zoom_in4_first == resolution_zoom

    label_placer = LabelPlacement(
        data=df,
        x='x',
        y='y',
        by='label',
        importance='importance',
        scale_function='asinh',
        tile_size=100,
    )

    labels = label_placer.compute()

    h0, zoom_in1_second = labels.iloc[0][['height', 'zoom_in']]
    h4, zoom_in4_second = labels.iloc[4][['height', 'zoom_in']]

    # Since we're using constant zoom scaling and the x position of the first
    # and fifth label perfectly overlap, the zoom level at which the collision
    # is resolved can be calculated as follows
    resolution_zoom = (h0 + h4) / 2 / 0.1
    z = np.float64(105.19451844298987)
    asinh_z = math.asinh(z) / math.asinh(1)

    assert (h0 + h4) / 2 * asinh_z <= 0.1 * z
    resolution_zoom = z

    assert zoom_in1_second < zoom_in4_second
    assert zoom_in4_second == resolution_zoom

    # The zoom level at which the collision is resolved must be higher for
    # asinh scaled labels vs labels drawn at a constant size
    assert zoom_in4_second > zoom_in4_first


def test_zoom_ranges():
    """Test that zoom ranges are properly applied and respected."""
    # Create a simple dataset with controlled positions
    df = pd.DataFrame(
        {
            'x': [0, 10, 20, 30, 40],
            'y': [0, 0, 0, 0, 0],
            'category': ['A', 'B', 'C', 'D', 'E'],
            'importance': [5, 4, 3, 2, 1],
        }
    )

    # Test 1: Basic zoom range application
    # Set different zoom ranges for different categories
    zoom_ranges = {
        'A': (2.0, 4.0),  # Only visible at zoom levels 2-4
        'B': (0.5, 3.0),  # Visible at zoom levels 0.5-3
        'C': (0.0, 1.5),  # Visible at zoom levels 0-1.5
        'D': (4.0, 8.0),  # Only visible at high zoom levels
        'E': (0.0, math.inf),  # Always visible above min zoom
    }

    # Convert zoom levels to zoom scales
    zoom_scales = {
        category: (
            2**min_level,
            2**max_level if not math.isinf(max_level) else math.inf,
        )
        for category, (min_level, max_level) in zoom_ranges.items()
    }

    label_placer = LabelPlacement(
        data=df,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        importance='importance',
        zoom_range=zoom_ranges,
    )

    result = label_placer.compute()

    # Check that zoom levels respect the specified ranges
    for _, row in result.iterrows():
        category = cast(str, row['label'])
        min_zoom_scale, max_zoom_scale = zoom_scales[category]

        # Check zoom_in respects minimum zoom constraint
        assert (
            row['zoom_in'] >= min_zoom_scale
        ), f'Label {category}: zoom_in {row["zoom_in"]} is less than min {min_zoom_scale}'

        # Check zoom_out respects maximum zoom constraint
        if not math.isinf(max_zoom_scale):
            assert (
                row['zoom_out'] <= max_zoom_scale
            ), f'Category {category}: zoom_out {row["zoom_out"]} is greater than max {max_zoom_scale}'

    # Test 2: Test collision resolution with zoom ranges
    # Create a dataset with overlapping labels but different zoom ranges
    df2 = pd.DataFrame(
        {
            'x': [0, 0, 0, 0],
            'y': [0, 0, 0, 0],
            'category': ['F', 'G', 'H', 'I'],
            'importance': [4, 3, 2, 1],
        }
    )

    # Define non-overlapping zoom ranges
    non_overlapping_ranges = {
        'F': (0.0, 2.0),  # Zoom levels 0-2
        'G': (2.0, 4.0),  # Zoom levels 2-4
        'H': (4.0, 6.0),  # Zoom levels 4-6
        'I': (6.0, 8.0),  # Zoom levels 6-8
    }

    label_placer2 = LabelPlacement(
        data=df2,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        importance='importance',
        zoom_range=non_overlapping_ranges,
        scale_function='asinh',
    )

    result2 = label_placer2.compute()

    # No collisions should occur since zoom ranges don't overlap
    # All labels should keep their original zoom ranges
    for _, row in result2.iterrows():
        category = cast(str, row['label'])
        min_level, max_level = non_overlapping_ranges[category]
        min_scale, max_scale = 2**min_level, 2**max_level

        assert (
            row['zoom_in'] == min_scale
        ), f'Category {category}: zoom_in was modified despite non-overlapping ranges'
        assert (
            row['zoom_out'] == max_scale
        ), f'Category {category}: zoom_out was modified despite non-overlapping ranges'

    label_placer3 = label_placer2.clone(scale_function='asinh')

    result3 = label_placer3.compute()

    # No collisions should occur since zoom ranges don't overlap
    # All labels should keep their original zoom ranges
    for _, row in result3.iterrows():
        category = cast(str, row['label'])
        min_level, max_level = non_overlapping_ranges[category]
        min_scale, max_scale = 2**min_level, 2**max_level

        assert (
            row['zoom_in'] == min_scale
        ), f'Category {category}: zoom_in was modified despite non-overlapping ranges'
        assert (
            row['zoom_out'] == max_scale
        ), f'Category {category}: zoom_out was modified despite non-overlapping ranges'

    # Test 3: Test overlapping zoom ranges with collision
    df4 = pd.DataFrame(
        {
            'x': [0, 0, 10, 10],  # Two sets of overlapping points
            'y': [0, 0.5, 10, 10.01],
            'category': ['J', 'K', 'L', 'M'],
            'importance': [4, 3, 2, 1],
        }
    )

    # Define overlapping zoom ranges
    overlapping_ranges = {
        'J': (0.0, 4.0),  # Zoom levels 0-4
        'K': (0.0, 4.0),  # Zoom levels 0-4 (overlaps with J)
        'L': (1.0, 5.0),  # Zoom levels 1-5
        'M': (1.0, 5.0),  # Zoom levels 1-5 (overlaps with L)
    }

    label_placer4 = LabelPlacement(
        data=df4,
        by='category',
        x='x',
        y='y',
        tile_size=100,
        importance='importance',
        zoom_range=overlapping_ranges,
        scale_function='constant',
    )

    result4 = label_placer4.compute()

    # Check that collision resolution respects zoom ranges
    # Lower priority labels (K, M) should have adjusted zoom_in values
    # Higher priority labels (J, L) should maintain their original zoom ranges

    # J should keep its original range
    j_row = result4[result4['label'] == 'J'].iloc[0]
    assert j_row['zoom_in'] == 2 ** overlapping_ranges['J'][0]
    assert j_row['zoom_out'] == 2 ** overlapping_ranges['J'][1]

    # K should have zoom_in adjusted due to collision with J
    k_row = result4[result4['label'] == 'K'].iloc[0]

    assert (
        k_row['zoom_in'] > 2 ** overlapping_ranges['K'][0]
    ), "K's zoom_in should be adjusted upward due to collision with J"
    assert (
        k_row['zoom_in'] <= 2 ** overlapping_ranges['K'][1]
    ), "K's zoom_in should still be within its max zoom range"

    # Similar checks for L and M
    l_row = result4[result4['label'] == 'L'].iloc[0]
    assert l_row['zoom_in'] == 2 ** overlapping_ranges['L'][0]

    m_row = result4[result4['label'] == 'M'].iloc[0]
    assert math.isinf(
        m_row['zoom_in']
    ), "M's zoom_in should be infinite due to the inability of resolving the collision in the zoom range"


def test_hierarchical_zoom_ranges(sample_data):
    """Test that zoom ranges are properly applied for different label types in a hierarchy."""
    # Test label type-specific zoom ranges
    type_zoom_ranges = {
        'category': (0.0, 2.0),  # Only visible at zoom levels 0-2
        'subcategory': (2.0, 4.0),  # Only visible at zoom levels 2-4
    }

    # Convert zoom levels to zoom scales for assertions
    type_zoom_scales = {
        label_type: (2**min_level, 2**max_level)
        for label_type, (min_level, max_level) in type_zoom_ranges.items()
    }

    label_placer = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        tile_size=50,
        hierarchical=True,
        zoom_range=type_zoom_ranges,
    )

    labels = label_placer.compute()

    # Verify that the zoom ranges are properly applied by label type
    for _, row in labels.iterrows():
        label_type = cast(str, row['label_type'])
        min_zoom_scale, max_zoom_scale = type_zoom_scales[label_type]

        # Check zoom_in respects minimum zoom constraint
        assert (
            row['zoom_in'] >= min_zoom_scale
        ), f'Label type "{label_type}": zoom_in {row["zoom_in"]} is less than min {min_zoom_scale}'

        # Check zoom_out respects maximum zoom constraint
        assert (
            row['zoom_out'] <= max_zoom_scale
        ), f'Label type "{label_type}": zoom_out {row["zoom_out"]} is greater than max {max_zoom_scale}'

    # Verify that categories and subcategories don't interfere with each other
    # since they have non-overlapping zoom ranges

    # All categories should be visible only up to zoom level 2
    category_rows = labels[labels['label_type'] == 'category']
    for _, row in category_rows.iterrows():
        assert (
            row['zoom_out'] <= type_zoom_scales['category'][1]
        ), f'Category "{row["label"]}" has zoom_out {row["zoom_out"]} exceeding max {type_zoom_scales["category"][1]}'

    # All subcategories should be visible only from zoom level 2 and up
    subcategory_rows = labels[labels['label_type'] == 'subcategory']
    for _, row in subcategory_rows.iterrows():
        assert (
            row['zoom_in'] >= type_zoom_scales['subcategory'][0]
        ), f'Subcategory {row["label"]} has zoom_in {row["zoom_in"]} less than min {type_zoom_scales["subcategory"][0]}'

    # Test for the transition between categories and subcategories
    # There should be a smooth transition at zoom level 2
    # Categories should fade out as subcategories fade in
    category_max_visible = max(category_rows['zoom_out'])
    subcategory_min_visible = min(subcategory_rows['zoom_in'])

    assert (
        abs(category_max_visible - subcategory_min_visible) < 1e-6
    ), f'There should be a smooth transition between categories and subcategories at zoom level 2'

    # Test combining label type zoom ranges with label-specific zoom ranges
    combined_zoom_ranges = {
        'category': (0.0, 2.0),  # Default for categories
        'subcategory': (2.0, 4.0),  # Default for subcategories
        'category:A': (0.0, 1.0),  # Specific override for category A
        'subcategory:A1': (1.0, 3.0),  # Specific override for subcategory A1
    }

    label_placer2 = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        tile_size=600,
        hierarchical=True,
        zoom_range=combined_zoom_ranges,
    )

    labels2 = label_placer2.compute()

    # Check that label-specific overrides are applied correctly
    cat_a_row = labels2[
        (labels2['label_type'] == 'category') & (labels2['label'] == 'A')
    ].iloc[0]
    assert (
        cat_a_row['zoom_out'] <= 2 ** combined_zoom_ranges['category:A'][1]
    ), f'Category A should have max zoom of 2**{combined_zoom_ranges["category:A"][1]}'

    # Find subcategory A1 if it exists
    a1_rows = labels2[
        (labels2['label_type'] == 'subcategory')
        & (labels2['label'].str.startswith('A1'))
    ]
    a1_row = a1_rows.iloc[0]
    assert (
        a1_row['zoom_in'] >= 2 ** combined_zoom_ranges['subcategory:A1'][0]
    ), f'Subcategory A1 should have min zoom of 2**{combined_zoom_ranges["subcategory:A1"][0]}'
    assert (
        a1_row['zoom_out'] <= 2 ** combined_zoom_ranges['subcategory:A1'][1]
    ), f'Subcategory A1 should have max zoom of 2**{combined_zoom_ranges["subcategory:A1"][1]}'


def test_exclude_labels(sample_data):
    """Test that labels can be excluded by type or specific label."""
    # Test excluding an entire label type
    label_placer_exclude_type = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        tile_size=600,
        exclude=['category'],
    )

    labels = label_placer_exclude_type.compute()

    # Should only have subcategory labels
    assert 'category' not in labels['label_type'].values
    assert 'subcategory' in labels['label_type'].values

    # Check that progress counter worked properly
    num_operations = label_placer_exclude_type._get_total_num_operations()
    # Should only count subcategory labels
    expected_num = len(sample_data['subcategory'].unique()) * 5
    assert num_operations == expected_num

    # Test excluding specific labels
    exclude_specific = ['category:A', 'subcategory:B1']  # Changed from dict

    label_placer_exclude_specific = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        tile_size=600,
        exclude=exclude_specific,
    )

    labels = label_placer_exclude_specific.compute()

    # Should exclude only specific labels
    assert 'A' not in labels[labels['label_type'] == 'category']['label'].values
    assert 'B' in labels[labels['label_type'] == 'category']['label'].values
    assert not any(
        label.startswith('B1')
        for label in labels[labels['label_type'] == 'subcategory']['label'].values
    )

    # Test that exclude properly filters labels in preprocessing
    label_placer_exclude_partial = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        exclude=['category:A', 'category:B'],
    )

    # Check the filtered data directly
    preprocessed = label_placer_exclude_partial._create_labels()

    # Should only contain categories C and D
    assert set(preprocessed['label']) == set(['C', 'D'])


def test_update_font_color_after_computing(sample_data):
    """Test updating font colors after computing labels."""
    label_placer = LabelPlacement(
        data=sample_data, by='category', x='x', y='y', color='red'
    )

    # Compute labels
    label_placer.compute()

    assert label_placer.labels is not None

    # Verify all labels use red color
    assert all(color == '#ff0000' for color in label_placer.color.values())
    assert all(
        row['font_color'] == '#ff0000' for _, row in label_placer.labels.iterrows()
    )

    # Update font color to blue
    label_placer.color = 'blue'

    # Verify all labels now use blue color
    assert all(color == '#0000ff' for color in label_placer.color.values())
    assert all(
        row['font_color'] == '#0000ff' for _, row in label_placer.labels.iterrows()
    )

    # Test updating with a dictionary
    color_map = {'category': 'green', 'category:A': 'yellow'}
    label_placer.color = color_map

    # Verify that colors are updated according to the map
    assert label_placer.color['category'] == '#008000'  # green
    assert label_placer.color['category:A'] == '#ffff00'  # yellow

    # Check that labels DataFrame is updated too
    a_labels = label_placer.labels[label_placer.labels['label'] == 'A']
    assert all(row['font_color'] == '#ffff00' for _, row in a_labels.iterrows())


def test_update_spatial_properties_before_computing(sample_data):
    """Test updating spatial properties before computing labels."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        size=12,
        zoom_range=(0.0, 2.0),
    )

    # Update spatial properties before computing
    label_placer.size = 16
    label_placer.zoom_range = {'category': (1.0, 3.0), 'category:A': (0.5, 2.5)}
    label_placer.positioning = 'highest_density'

    # Compute labels
    labels = label_placer.compute()

    # Verify properties were applied
    assert label_placer.size['category'] == 16
    assert label_placer.zoom_range['category'] == (1.0, 3.0)
    assert label_placer.zoom_range['category:A'] == (0.5, 2.5)
    assert label_placer.positioning == 'highest_density'

    # Check that computed labels reflect these settings
    a_labels = labels[labels['label'] == 'A']
    assert all(row['font_size'] == 16 for _, row in a_labels.iterrows())
    assert all(row['zoom_in'] >= np.float64(2**0.5) for _, row in a_labels.iterrows())


def test_update_spatial_properties_after_computing(sample_data):
    """Test that updating spatial properties after computing raises an error."""
    label_placer = LabelPlacement(data=sample_data, by='category', x='x', y='y')

    # Compute labels
    label_placer.compute()

    # Attempting to update spatial properties should raise ValueError
    with pytest.raises(
        ValueError, match='Cannot update font sizes after having computed labels'
    ):
        label_placer.size = 16

    with pytest.raises(
        ValueError, match='Cannot update zoom ranges after having computed labels'
    ):
        label_placer.zoom_range = (1.0, 3.0)

    with pytest.raises(
        ValueError,
        match='Cannot update positioning method after having computed labels',
    ):
        label_placer.positioning = 'highest_density'

    with pytest.raises(
        ValueError, match='Cannot update font faces after having computed labels'
    ):
        label_placer.font = arial.bold


def test_reset_for_updating_spatial_properties(sample_data):
    """Test resetting a LabelPlacement instance to update spatial properties."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        size=12,
        positioning='center_of_mass',
    )

    # Compute labels with initial settings
    initial_labels = label_placer.compute()

    # Store some values for comparison
    initial_zoom_in = initial_labels['zoom_in'].values.copy()

    assert isinstance(initial_zoom_in, np.ndarray)

    # Reset the instance
    label_placer.reset()

    # Verify that labels are cleared
    assert label_placer.labels is None
    assert label_placer.tiles is None
    assert not label_placer.computed

    # Now we can update spatial properties
    label_placer.size = 24
    label_placer.positioning = 'highest_density'

    # Recompute with new settings
    new_labels = label_placer.compute()

    # Verify new settings were applied
    assert all(row['font_size'] == 24 for _, row in new_labels.iterrows())

    # Positioning change should result in different label positions
    new_zoom_in = new_labels['zoom_in'].values

    assert isinstance(new_zoom_in, np.ndarray)

    assert not np.array_equal(new_zoom_in, initial_zoom_in)


def test_clone_for_updating_spatial_properties(sample_data):
    """Test cloning a LabelPlacement instance to update spatial properties."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        size=12,
        zoom_range=(0.0, 2.0),
    )

    # Compute labels with initial settings
    initial_labels = label_placer.compute()

    # Clone the instance with different spatial properties
    cloned_placer = label_placer.clone(
        size=18, zoom_range=(1.0, 3.0), positioning='highest_density'
    )

    # Verify original instance is unchanged
    assert label_placer.size['category'] == 12
    assert label_placer.zoom_range['category'][0] == 0.0
    assert label_placer.zoom_range['category'][1] == 2.0

    # Verify cloned instance has new properties
    assert cloned_placer.size['category'] == 18
    assert cloned_placer.zoom_range['category'][0] == 1.0
    assert cloned_placer.zoom_range['category'][1] == 3.0
    assert cloned_placer.positioning == 'highest_density'

    # Compute labels for cloned instance
    cloned_labels = cloned_placer.compute()

    # Verify computed labels reflect new settings
    assert all(row['font_size'] == 18 for _, row in cloned_labels.iterrows())
    assert all(
        row['zoom_in'] >= np.float64(2**1.0) for _, row in cloned_labels.iterrows()
    )

    # Verify that changing the clone doesn't affect the original
    assert initial_labels is not None
    assert label_placer.labels is initial_labels


def test_reset_and_multiple_property_changes(sample_data):
    """Test resetting and changing multiple properties before recomputing."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        size=12,
        zoom_range=(0.0, 2.0),
        color='red',
    )

    # Compute labels with initial settings
    label_placer.compute()

    # Change a non-spatial property - should work without reset
    label_placer.color = 'blue'

    # Reset to change spatial properties
    label_placer.reset()

    # Change multiple properties
    label_placer.size = 20
    label_placer.zoom_range = (1.0, 4.0)
    label_placer.positioning = 'highest_density'
    label_placer.exclude = ['category:A']

    # Recompute
    new_labels = label_placer.compute()

    # Verify spatial properties were applied
    assert label_placer.size['category'] == 20
    assert label_placer.zoom_range['category'][0] == 1.0
    assert label_placer.zoom_range['category'][1] == 4.0

    # Verify non-spatial property was maintained
    assert all(
        color == '#0000ff'
        for color in label_placer.color.values()
        if not ':' in color or not color.endswith(':A')
    )

    category_labels = new_labels[new_labels['label_type'] == 'category']['label'].values

    assert isinstance(category_labels, np.ndarray)
    assert all(isinstance(label, str) for label in category_labels)

    # Verify exclusion was applied
    assert 'A' not in category_labels


def test_complex_workflow_with_reset_and_clone(sample_data):
    """Test a complex workflow with reset and clone operations."""
    # Create initial instance with basic settings
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        size=12,
        positioning='center_of_mass',
        scale_function='asinh',
    )

    # Compute initial labels
    initial_labels = label_placer.compute()

    # Clone with some different settings for comparison
    comparison_placer = label_placer.clone(size=16)
    comparison_labels = comparison_placer.compute()

    # Reset original and modify settings
    label_placer.reset()
    label_placer.size = 20

    # Recompute with new settings
    new_labels = label_placer.compute()

    # Verify three different sets of results
    assert all(row['font_size'] == 12 for _, row in initial_labels.iterrows())
    assert all(row['font_size'] == 16 for _, row in comparison_labels.iterrows())
    assert all(row['font_size'] == 20 for _, row in new_labels.iterrows())

    # Verify different positioning resulted in different label placements
    initial_zoom_in = initial_labels[['zoom_in']].values
    comparison_zoom_in = comparison_labels[['zoom_in']].values
    new_zoom_in = new_labels[['zoom_in']].values

    assert isinstance(initial_zoom_in, np.ndarray)
    assert isinstance(comparison_zoom_in, np.ndarray)
    assert isinstance(new_zoom_in, np.ndarray)

    # At least some zoom_in should be different
    assert not np.array_equal(initial_zoom_in, comparison_zoom_in)
    assert not np.array_equal(initial_zoom_in, new_zoom_in)
    assert not np.array_equal(comparison_zoom_in, new_zoom_in)


def test_update_data_and_columns_with_label_constraints(sample_data):
    """Test updating data and columns with the same set of unique labels."""
    label_placer = LabelPlacement(data=sample_data, by='category', x='x', y='y')

    # Create a modified dataset with the SAME unique labels but different points
    modified_data = sample_data.copy()
    # Double all the x and y values
    modified_data['x'] = modified_data['x'] * 2
    modified_data['y'] = modified_data['y'] * 2

    # Should succeed because labels are the same
    label_placer.data = modified_data

    # Verify the update happened
    assert np.isclose(label_placer._x_max, sample_data['x'].max() * 2)
    assert np.isclose(label_placer._y_max, sample_data['y'].max() * 2)

    # Create data with different labels
    new_categories_data = sample_data.copy()
    new_categories_data.loc[0:10, 'category'] = 'NEW_CATEGORY'

    # Attempt to update with data that has different labels
    with pytest.raises(
        ValueError,
        match=re.escape(
            'Cannot update data as it would result in different unique labels. '
            'Use clone() or create a new instance instead.'
        ),
    ):
        label_placer.data = new_categories_data

    # Test updating 'by' columns with the same labels
    # Create new column that has the same unique values as 'category'
    modified_data['category_copy'] = modified_data['category']

    # Should succeed because the unique labels are the same
    label_placer.by = 'category_copy'

    # Verify the update happened
    assert label_placer.by == ['category_copy']

    # Create data with a column that would result in different labels
    modified_data['different_categories'] = modified_data['category'] + '_suffix'

    # Attempt to update 'by' with a column that would result in different labels
    with pytest.raises(
        ValueError,
        match=re.escape(
            'Cannot update "by" columns as it would result in different unique labels. '
            'Use clone() or create a new instance instead.'
        ),
    ):
        label_placer.by = 'different_categories'


def test_style_adjustments_with_clone(sample_data):
    """Test creating variations of style properties with clone."""
    base_labeler = LabelPlacement(
        data=sample_data, by='category', x='x', y='y', size=12, color='black'
    )

    # Create style variations with clone
    large_labeler = base_labeler.clone(size=24)
    small_labeler = base_labeler.clone(size=8)
    colored_labeler = base_labeler.clone(
        color={'category': 'blue', 'category:A': 'red', 'category:B': 'green'}
    )

    # Compute all variations
    base_labels = base_labeler.compute()
    large_labels = large_labeler.compute()
    small_labels = small_labeler.compute()
    colored_labels = colored_labeler.compute()

    # Verify font sizes
    assert all(row['font_size'] == 12 for _, row in base_labels.iterrows())
    assert all(row['font_size'] == 24 for _, row in large_labels.iterrows())
    assert all(row['font_size'] == 8 for _, row in small_labels.iterrows())

    # Verify colors in colored variant
    assert colored_labeler.color['category'] == '#0000ff'  # blue
    assert colored_labeler.color['category:A'] == '#ff0000'  # red
    assert colored_labeler.color['category:B'] == '#008000'  # green

    # Check label DataFrame colors
    a_labels = colored_labels[colored_labels['label'] == 'A']
    b_labels = colored_labels[colored_labels['label'] == 'B']
    other_labels = colored_labels[~colored_labels['label'].isin(['A', 'B'])]

    assert all(row['font_color'] == '#ff0000' for _, row in a_labels.iterrows())
    assert all(row['font_color'] == '#008000' for _, row in b_labels.iterrows())
    assert all(row['font_color'] == '#0000ff' for _, row in other_labels.iterrows())


def test_error_handling_with_property_updates():
    """Test error handling when updating properties with invalid values."""

    df = pd.DataFrame(
        {'x': [0.1, 0.2, 0.3], 'y': [0.1, 0.2, 0.3], 'category': ['A', 'B', 'C']}
    )

    label_placer = LabelPlacement(data=df, by='category', x='x', y='y')

    # Test updating with invalid column name
    with pytest.raises(ValueError, match="Column 'invalid_column' not found in data."):
        label_placer.x = 'invalid_column'

    with pytest.raises(
        ValueError, match=re.escape("Columns not found in data: ['invalid_column']")
    ):
        label_placer.by = 'invalid_column'

    with pytest.raises(
        ValueError, match=re.escape("Columns not found in data: ['invalid_column']")
    ):
        label_placer.by = ['category', 'invalid_column']

    with pytest.raises(ValueError, match="Column 'invalid_column' not found in data."):
        label_placer.importance = 'invalid_column'

    # Compute labels
    label_placer.compute()

    # Test attempting to update data after computing
    new_df = pd.DataFrame(
        {'x': [0.4, 0.5, 0.6], 'y': [0.4, 0.5, 0.6], 'category': ['D', 'E', 'F']}
    )

    with pytest.raises(
        ValueError, match='Cannot update data after having computed labels'
    ):
        label_placer.data = new_df


def test_hierarchy_validation_with_excluded_values():
    """Test that hierarchy validation works correctly with excluded values."""
    # Create a dataset with hierarchy violations
    df = pd.DataFrame(
        {
            'x': [0, 1, 2, 3, 4, 5, 6],
            'y': [0, 1, 2, 3, 4, 5, 6],
            'label1': [None, 'A', 'A', 'A', 'B', 'B', 'B'],
            'label2': [None, None, 'A1', 'A2', None, 'B1', 'B2'],
        }
    )

    # Verify that None values don't cause a hierarchy violation
    try:
        # This should not raise a ValueError
        label_placer = LabelPlacement(
            data=df,
            by=['label1', 'label2'],
            x='x',
            y='y',
            hierarchical=True,
        )
        # If we get here, no error was raised
        assert True
    except ValueError as e:
        if 'Hierarchy violation' in str(e):
            pytest.fail(f'Hierarchy validation failed with None values: {e}')
        else:
            # Other ValueError types should still be raised
            raise

    df['label1'] = df['label1'].astype('category')
    df['label2'] = df['label2'].astype('category')

    # Verify that NaN values don't cause a hierarchy violation
    try:
        # This should not raise a ValueError
        label_placer = LabelPlacement(
            data=df,
            by=['label1', 'label2'],
            x='x',
            y='y',
            hierarchical=True,
        )
        # If we get here, no error was raised
        assert True
    except ValueError as e:
        if 'Hierarchy violation' in str(e):
            pytest.fail(f'Hierarchy validation failed with NaN values: {e}')
        else:
            # Other ValueError types should still be raised
            raise

    df = pd.DataFrame(
        {
            'x': [0, 1, 2, 3, 4, 5, 6],
            'y': [0, 1, 2, 3, 4, 5, 6],
            'label1': ['None', 'A', 'A', 'A', 'B', 'B', 'B'],
            'label2': ['None', 'None', 'A1', 'A2', 'None', 'B1', 'B2'],
        }
    )

    # Without excluding anything, the validation should fail
    with pytest.raises(ValueError, match='Hierarchy violation'):
        LabelPlacement(
            data=df,
            by=['label1', 'label2'],
            x='x',
            y='y',
            hierarchical=True,
        )

    # By excluding the problematic None values, the validation should pass
    label_placer = LabelPlacement(
        data=df,
        by=['label1', 'label2'],
        x='x',
        y='y',
        hierarchical=True,
        exclude=['label1:None', 'label2:None'],  # Exclude None values
    )

    # Make sure the instance was created successfully
    assert label_placer.by == ['label1', 'label2']
    assert label_placer.hierarchical == True

    # Compute labels to ensure everything works
    labels = label_placer.compute()

    # The None values should be excluded
    assert 'None' not in labels[labels['label_type'] == 'label1']['label'].values
    assert 'None' not in labels[labels['label_type'] == 'label2']['label'].values

    # Only valid hierarchical labels should remain
    expected_label1_values = {'A', 'B'}
    expected_label2_values = {'A1', 'A2', 'B1', 'B2'}

    actual_label1_values = set(labels[labels['label_type'] == 'label1']['label'].values)
    actual_label2_values = set(labels[labels['label_type'] == 'label2']['label'].values)

    assert actual_label1_values == expected_label1_values
    assert actual_label2_values == expected_label2_values


def test_chunking(sample_data):
    """Test the validity of label with data chunking."""

    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        importance='importance',
    )
    labels = label_placer.compute()

    # Check that labels are sorted descendingly by importance
    assert labels['importance'].is_monotonic_decreasing

    chunked_label_placer = label_placer.clone()
    chunked_labels = chunked_label_placer.compute(chunk_size=len(sample_data) // 5)

    # Check that labels are sorted descendingly by importance
    assert chunked_labels['importance'].is_monotonic_decreasing


def test_large_scale_performance():
    """Test performance with a large dataset of 20,000 points."""
    # Create a larger dataset
    np.random.seed(42)
    n = 20000

    # Generate random coordinates
    large_df = pd.DataFrame(
        {
            'x': np.random.uniform(0, 1, n),
            'y': np.random.uniform(0, 1, n),
            'category': np.random.choice(
                ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], n
            ),
            'subcategory': [''] * n,
            'importance': np.random.uniform(0, 1, n),
        }
    )

    # Ensure subcategories are properly nested within categories
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for category in categories:
        df_cat = large_df['category'] == category
        subcategories = [f'{category}{s}' for s in ['1', '2', '3']]
        large_df.loc[df_cat, 'subcategory'] = np.random.choice(
            subcategories, len(large_df[df_cat])
        )

    # Time the initialization
    import time

    label_placer = LabelPlacement(
        data=large_df,
        by='category',
        x='x',
        y='y',
        tile_size=600,
        importance='importance',
    )

    # Time the computation
    start_time = time.time()
    labels = label_placer.compute(chunk_size=5000)
    compute_time = time.time() - start_time

    # Basic checks to verify functionality
    assert label_placer.labels is not None
    assert label_placer.tiles is not None
    assert len(label_placer.labels) <= 10  # Should have at most 10 category labels

    # Basic performance check
    assert compute_time < 5.0, f'Computation took too long: {compute_time} seconds'

    # Verify the hierarchical case as well
    start_time = time.time()
    hierarchical_placer = LabelPlacement(
        data=large_df,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        tile_size=600,
        importance='importance',
        hierarchical=True,
    )
    hier_labels = hierarchical_placer.compute(chunk_size=5000)
    hier_compute_time = time.time() - start_time

    assert hier_compute_time < 5.0, f'Computation took too long: {compute_time} seconds'

    assert hierarchical_placer.labels is not None
    assert hierarchical_placer.tiles is not None
    assert (
        len(hierarchical_placer.labels) <= 40
    )  # Should have at most 10 categories + 30 subcategories

    # Verify the label types are correctly ordered for hierarchical labels
    assert all(
        hier_labels.groupby('label_type', observed=True)[
            'importance'
        ].is_monotonic_decreasing
    )


def test_point_label_column_handling(sample_data):
    """Test handling of point label columns (columns with exclamation mark)."""

    data = sample_data.copy()
    data['id'] = range(len(data))

    # Test with a single point label column
    point_labeler = LabelPlacement(
        data=data,
        by='category!',
        x='x',
        y='y',
    )

    assert point_labeler.by == ['category']
    assert point_labeler._point_label_columns == ['category']

    point_labels = point_labeler.compute()

    assert len(point_labels) == len(data)

    # Test with multiple columns including one point label
    mixed_labeler = LabelPlacement(data=data, by=['category', 'id!'], x='x', y='y')

    assert mixed_labeler.by == ['category', 'id']
    assert mixed_labeler._point_label_columns == ['id']

    # Compute and check that results include both category labels and point labels
    mixed_labels = mixed_labeler.compute()
    assert 'category' in mixed_labels['label_type'].values
    assert 'id' in mixed_labels['label_type'].values

    # Test with multiple point label columns (should use only the first one)
    multi_point_labeler = LabelPlacement(
        data=data, by=['category!', 'subcategory!'], x='x', y='y'
    )

    assert multi_point_labeler.by == ['category', 'subcategory']
    assert multi_point_labeler._point_label_columns == [
        'category'
    ]  # Only first one used

    multi_point_labels = multi_point_labeler.compute()

    # Ensure only the first point label column was used for individual labels
    assert len(multi_point_labels) == len(data) + data['subcategory'].nunique()

    # For hierarchical data with point labels, check that a non-hierarchical
    # column with an exclamation mark does not break the data validation
    np.random.seed(42)
    nonhierarchical_data = data.copy()
    nonhierarchical_data['category2'] = np.random.choice(['I', 'II', 'III'], len(data))

    # Verify that the data is truly non-hierarchical
    with pytest.raises(ValueError, match='Hierarchy violation'):
        LabelPlacement(
            data=nonhierarchical_data,
            by=['category', 'category2'],
            x='x',
            y='y',
            hierarchical=True,
        )

    # The exclamation mark should make the hierarchal validation pass
    LabelPlacement(
        data=nonhierarchical_data,
        by=['category', 'category2!'],
        x='x',
        y='y',
        hierarchical=True,
    )
