import pytest
from test_label_placement import sample_data

from jscatter.label_placement.label_placement import LabelPlacement


def test_background_property(sample_data):
    """Test the background property getter and setter."""
    # Test with initial background color
    label_placer = LabelPlacement(
        data=sample_data, by='category', x='x', y='y', background='black'
    )

    assert label_placer.background == 'black'

    # Test setting background after initialization
    label_placer.background = 'blue'
    assert label_placer.background == 'blue'

    # Test with RGB tuple
    label_placer.background = (0.5, 0.5, 0.5)
    assert label_placer.background == (0.5, 0.5, 0.5)

    # Test that background can be changed after compute
    label_placer.compute()
    label_placer.background = 'red'
    assert label_placer.background == 'red'

    # Test how background affects default colors
    black_bg_placer = LabelPlacement(
        data=sample_data, by='category', x='x', y='y', background='black'
    )

    white_bg_placer = LabelPlacement(
        data=sample_data, by='category', x='x', y='y', background='white'
    )

    # Default color should be white for black background
    assert black_bg_placer.color['category'] == '#ffffff'
    # Default color should be black for white background
    assert white_bg_placer.color['category'] == '#000000'


def test_font_setter_before_compute(sample_data):
    """Test setting font properties before computing labels."""
    from jscatter.font import arial

    label_placer = LabelPlacement(
        data=sample_data, by='category', x='x', y='y', font=arial.regular
    )

    # Check initial font
    assert label_placer.font['category'] == arial.regular

    # Update font before compute
    label_placer.font = arial.bold
    assert label_placer.font['category'] == arial.bold

    # Update with different fonts for different categories
    label_placer.font = {'category': arial.bold, 'category:A': arial.italic}

    assert label_placer.font['category'] == arial.bold
    assert label_placer.font['category:A'] == arial.italic

    # Test font setting with multiple label types
    label_placer = LabelPlacement(
        data=sample_data, by=['category', 'subcategory'], x='x', y='y'
    )

    label_placer.font = [arial.bold, arial.italic]
    assert label_placer.font['category'] == arial.bold
    assert label_placer.font['subcategory'] == arial.italic

    # Compute and verify these fonts are used
    labels = label_placer.compute()

    cat_labels = labels[labels['label_type'] == 'category']
    subcat_labels = labels[labels['label_type'] == 'subcategory']

    assert all(row['font_face'] == arial.bold.face for _, row in cat_labels.iterrows())
    assert all(
        row['font_style'] == arial.bold.style for _, row in cat_labels.iterrows()
    )
    assert all(
        row['font_weight'] == arial.bold.weight for _, row in cat_labels.iterrows()
    )

    assert all(
        row['font_face'] == arial.italic.face for _, row in subcat_labels.iterrows()
    )
    assert all(
        row['font_style'] == arial.italic.style for _, row in subcat_labels.iterrows()
    )
    assert all(
        row['font_weight'] == arial.italic.weight for _, row in subcat_labels.iterrows()
    )


def test_exclude_getter(sample_data):
    """Test the exclude property getter."""
    exclude_list = ['category:A', 'subcategory:B1']

    label_placer = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        exclude=exclude_list,
    )

    # Check that the getter returns the correct list of excluded items
    retrieved_exclude = label_placer.exclude

    assert isinstance(retrieved_exclude, list)
    assert set(retrieved_exclude) == set(exclude_list)

    # Test with an entire column excluded
    label_placer = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        exclude=['category'],
    )

    assert label_placer.exclude == ['category']

    # Test with empty exclude list
    label_placer = LabelPlacement(
        data=sample_data, by=['category', 'subcategory'], x='x', y='y', exclude=[]
    )

    assert label_placer.exclude == []

    # Test exclude list after computation
    label_placer.compute()
    assert set(label_placer.exclude) == set([])


def test_x_y_setters(sample_data):
    """Test the x and y coordinate column name setters."""
    # Create a copy of sample_data with alternate column names
    alt_data = sample_data.copy()
    alt_data['x_alt'] = alt_data['x']
    alt_data['y_alt'] = alt_data['y']

    label_placer = LabelPlacement(data=alt_data, by='category', x='x', y='y')

    # Store original coordinate extents
    orig_x_min = label_placer._x_min
    orig_x_max = label_placer._x_max
    orig_y_min = label_placer._y_min
    orig_y_max = label_placer._y_max

    # Change x and y columns
    label_placer.x = 'x_alt'
    label_placer.y = 'y_alt'

    # Check that getters work
    assert label_placer.x == 'x_alt'
    assert label_placer.y == 'y_alt'

    # Check that coordinate extents were updated
    assert label_placer._x_min == orig_x_min
    assert label_placer._x_max == orig_x_max
    assert label_placer._y_min == orig_y_min
    assert label_placer._y_max == orig_y_max

    # Test with non-existent columns
    with pytest.raises(ValueError, match="Column 'nonexistent' not found in data."):
        label_placer.x = 'nonexistent'

    with pytest.raises(ValueError, match="Column 'nonexistent' not found in data."):
        label_placer.y = 'nonexistent'

    # Test that x/y setters cannot be used after compute
    label_placer.compute()

    with pytest.raises(
        ValueError,
        match='Cannot update x-coordinate column after having computed labels',
    ):
        label_placer.x = 'x'

    with pytest.raises(
        ValueError,
        match='Cannot update y-coordinate column after having computed labels',
    ):
        label_placer.y = 'y'


def test_hierarchical_setter(sample_data):
    """Test the hierarchical property setter."""
    label_placer = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory'],
        x='x',
        y='y',
        hierarchical=False,
    )

    # Check initial value
    assert label_placer.hierarchical == False

    # Update the value
    label_placer.hierarchical = True
    assert label_placer.hierarchical == True

    # Update back to False
    label_placer.hierarchical = False
    assert label_placer.hierarchical == False

    # Test that hierarchical flag affects size defaults
    hierarchical_placer = LabelPlacement(
        data=sample_data,
        by=['category', 'subcategory', 'id'],
        x='x',
        y='y',
        hierarchical=True,
    )

    # Check that sizes decrease by hierarchy level when hierarchical=True
    assert (
        hierarchical_placer.size['category'] > hierarchical_placer.size['subcategory']
    )
    assert hierarchical_placer.size['subcategory'] > hierarchical_placer.size['id']

    # Test that hierarchical cannot be changed after compute
    label_placer.compute()

    with pytest.raises(
        ValueError,
        match='Cannot update hierarchical setting after having computed labels',
    ):
        label_placer.hierarchical = True


def test_importance_setter(sample_data):
    """Test the importance column name setter."""
    label_placer = LabelPlacement(
        data=sample_data, by='category', x='x', y='y', importance=None
    )

    # Check initial value
    assert label_placer.importance is None

    # Set to a valid column
    label_placer.importance = 'importance'
    assert label_placer.importance == 'importance'

    # Set back to None
    label_placer.importance = None
    assert label_placer.importance is None

    # Test with invalid column name
    with pytest.raises(ValueError, match="Column 'nonexistent' not found in data."):
        label_placer.importance = 'nonexistent'

    # Test that importance cannot be changed after compute
    label_placer.compute()

    with pytest.raises(
        ValueError, match='Cannot update importance column after having computed labels'
    ):
        label_placer.importance = 'importance'


def test_importance_aggregation_getter_setter(sample_data):
    """Test the importance_aggregation getter and setter."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        importance='importance',
        importance_aggregation='mean',
    )

    # Check initial value
    assert label_placer.importance_aggregation == 'mean'

    # Update to different valid aggregation methods
    for method in ['min', 'max', 'median', 'sum']:
        label_placer.importance_aggregation = method
        assert label_placer.importance_aggregation == method

    # Test with invalid method
    with pytest.raises(Exception):  # Type of exception depends on implementation
        label_placer.importance_aggregation = 'invalid_method'
        label_placer.compute()

    label_placer.importance_aggregation = 'mean'

    # Test that importance_aggregation cannot be changed after compute
    label_placer.compute()

    with pytest.raises(
        ValueError,
        match='Cannot update importance aggregation methid after having computed labels',
    ):
        label_placer.importance_aggregation = 'median'


def test_bbox_percentile_range_setter(sample_data):
    """Test the bbox_percentile_range setter."""
    label_placer = LabelPlacement(
        data=sample_data, by='category', x='x', y='y', bbox_percentile_range=(5, 95)
    )

    # Check initial value
    assert label_placer.bbox_percentile_range == (5, 95)

    # Update to different range
    label_placer.bbox_percentile_range = (10, 90)
    assert label_placer.bbox_percentile_range == (10, 90)

    # Update to extreme range (include all points)
    label_placer.bbox_percentile_range = (0, 100)
    assert label_placer.bbox_percentile_range == (0, 100)

    # Test with invalid ranges - implementation may or may not validate these
    try:
        label_placer.bbox_percentile_range = (95, 5)  # Reversed
        label_placer.bbox_percentile_range = (-5, 105)  # Out of bounds
        label_placer.compute()
    except Exception:
        pass

    label_placer.bbox_percentile_range = (5, 95)

    # Test that bbox_percentile_range cannot be changed after compute
    label_placer.compute()

    with pytest.raises(
        ValueError,
        match='Cannot update bbox percentile range after having computed labels',
    ):
        label_placer.bbox_percentile_range = (25, 75)


def test_tile_size_setter(sample_data):
    """Test the tile_size setter."""
    label_placer = LabelPlacement(
        data=sample_data, by='category', x='x', y='y', tile_size=256
    )

    # Check initial value
    assert label_placer.tile_size == 256

    # Update tile size
    label_placer.tile_size = 512
    assert label_placer.tile_size == 512

    # Update to very small tile size
    label_placer.tile_size = 64
    assert label_placer.tile_size == 64

    # Test that tile_size cannot be changed after compute
    label_placer.compute()

    with pytest.raises(
        ValueError, match='Cannot update tile size after having computed labels'
    ):
        label_placer.tile_size = 1024


def test_max_labels_per_tile_setter(sample_data):
    """Test the max_labels_per_tile setter."""
    label_placer = LabelPlacement(
        data=sample_data, by='category', x='x', y='y', max_labels_per_tile=50
    )

    # Check initial value
    assert label_placer.max_labels_per_tile == 50

    # Update max labels per tile
    label_placer.max_labels_per_tile = 100
    assert label_placer.max_labels_per_tile == 100

    # Update to unlimited (0)
    label_placer.max_labels_per_tile = 0
    assert label_placer.max_labels_per_tile == 0

    # Test that max_labels_per_tile cannot be changed after compute
    label_placer.compute()

    with pytest.raises(
        ValueError,
        match='Cannot update maximum labels per tile after having computed labels',
    ):
        label_placer.max_labels_per_tile = 25


def test_target_aspect_ratio_getter_setter(sample_data):
    """Test the target_aspect_ratio getter and setter."""
    label_placer = LabelPlacement(
        data=sample_data, by='category', x='x', y='y', target_aspect_ratio=None
    )

    # Check initial value
    assert label_placer.target_aspect_ratio is None

    # Update to a specific ratio
    label_placer.target_aspect_ratio = 1.5  # Width = 1.5 * height
    assert label_placer.target_aspect_ratio == 1.5

    # Update to square ratio
    label_placer.target_aspect_ratio = 1.0
    assert label_placer.target_aspect_ratio == 1.0

    # Update back to None (disable line break optimization)
    label_placer.target_aspect_ratio = None
    assert label_placer.target_aspect_ratio is None

    # Test that target_aspect_ratio cannot be changed after compute
    label_placer.compute()

    with pytest.raises(
        ValueError,
        match='Cannot update target aspect ratio after having computed labels',
    ):
        label_placer.target_aspect_ratio = 2.0


def test_max_lines_getter_setter(sample_data):
    """Test the max_lines getter and setter."""
    label_placer = LabelPlacement(
        data=sample_data,
        by='category',
        x='x',
        y='y',
        target_aspect_ratio=1.5,
        max_lines=None,
    )

    # Check initial value
    assert label_placer.max_lines is None

    # Update to a specific value
    label_placer.max_lines = 3
    assert label_placer.max_lines == 3

    # Update to unlimited lines
    label_placer.max_lines = None
    assert label_placer.max_lines is None

    # Test that max_lines cannot be changed after compute
    label_placer.compute()

    with pytest.raises(
        ValueError, match='Cannot update maximum lines after having computed labels'
    ):
        label_placer.max_lines = 2


def test_verbosity_getter_setter(sample_data):
    """Test the verbosity getter and setter."""
    label_placer = LabelPlacement(
        data=sample_data, by='category', x='x', y='y', verbosity='warning'
    )

    # Check initial value
    assert label_placer.verbosity == 'warning'

    # Test with all valid verbosity levels
    for level in ['debug', 'info', 'warning', 'error', 'critical']:
        label_placer.verbosity = level
        assert label_placer.verbosity == level

    # Test that verbosity can be changed after compute
    label_placer.compute()
    label_placer.verbosity = 'debug'
    assert label_placer.verbosity == 'debug'


def test_scale_function_setter(sample_data):
    """Test the scale_function setter."""
    label_placer = LabelPlacement(
        data=sample_data, by='category', x='x', y='y', scale_function='constant'
    )

    # Check initial value
    assert label_placer.scale_function == 'constant'

    # Update to different scale function
    label_placer.scale_function = 'asinh'
    assert label_placer.scale_function == 'asinh'

    # Update back to constant
    label_placer.scale_function = 'constant'
    assert label_placer.scale_function == 'constant'

    # Test that scale_function cannot be changed after compute
    label_placer.compute()

    with pytest.raises(
        ValueError, match='Cannot update scale function after having computed labels'
    ):
        label_placer.scale_function = 'asinh'
