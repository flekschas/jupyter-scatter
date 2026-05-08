import numpy as np
import pandas as pd
import pytest

from functools import partial
from matplotlib.colors import AsinhNorm, LogNorm, Normalize, PowerNorm, SymLogNorm

from jscatter.jscatter import Scatter, component_idx_to_name, check_encoding_dtype
from jscatter.utils import create_default_norm, to_ndc, TimeNormalize


@pytest.fixture
def df() -> pd.DataFrame:
    num_groups = 8

    data = np.zeros((500, 7))
    data[:, 0] = np.linspace(0, 1, 500)
    data[:, 1] = np.linspace(0, 1, 500)
    data[:, 2] = np.random.rand(500) * 100
    data[:, 3] = (np.random.rand(500) * 100).astype(int)
    data[:, 4] = np.round(np.random.rand(500) * (num_groups - 1)).astype(int)
    data[:, 5] = np.repeat(np.arange(100), 5).astype(int)
    data[:, 6] = np.resize(np.arange(5), 500).astype(int)

    df = pd.DataFrame(
        data, columns=['a', 'b', 'c', 'd', 'group', 'connect', 'connect_order']
    )
    df['group'] = (
        df['group']
        .astype('int')
        .astype('category')
        .map(lambda c: chr(65 + c), na_action=None)
    )
    df['connect'] = df['connect'].astype('int')
    df['connect_order'] = df['connect_order'].astype('int')

    return df


@pytest.fixture
def df2() -> pd.DataFrame:
    num_groups = 10

    data = np.zeros((500, 7))
    data[:, 0] = np.linspace(-2, 2, 500)
    data[:, 1] = np.linspace(-2, 2, 500)
    data[:, 2] = np.random.rand(500) * 200
    data[:, 3] = (np.random.rand(500) * 200).astype(int)
    data[:, 4] = np.round(np.random.rand(500) * (num_groups - 1)).astype(int)
    data[:, 5] = np.repeat(np.arange(100), 5).astype(int)
    data[:, 6] = np.resize(np.arange(5), 500).astype(int)

    df = pd.DataFrame(
        data, columns=['a', 'b', 'c', 'd', 'group', 'connect', 'connect_order']
    )
    df['group'] = (
        df['group']
        .astype('int')
        .astype('category')
        .map(lambda c: chr(65 + c), na_action=None)
    )
    df['connect'] = df['connect'].astype('int')
    df['connect_order'] = df['connect_order'].astype('int')

    return df


@pytest.fixture
def df3() -> pd.DataFrame:
    num_groups = 8

    data = np.zeros((1000, 7))
    data[:, 0] = np.linspace(0, 1, 1000)
    data[:, 1] = np.linspace(0, 1, 1000)
    data[:, 2] = np.random.rand(1000) * 100
    data[:, 3] = (np.random.rand(1000) * 100).astype(int)
    data[:, 4] = np.round(np.random.rand(1000) * (num_groups - 1)).astype(int)
    data[:, 5] = np.repeat(np.arange(100), 10).astype(int)
    data[:, 6] = np.resize(np.arange(5), 1000).astype(int)

    df = pd.DataFrame(
        data, columns=['a', 'b', 'c', 'd', 'group', 'connect', 'connect_order']
    )
    df['group'] = (
        df['group']
        .astype('int')
        .astype('category')
        .map(lambda c: chr(65 + c), na_action=None)
    )
    df['connect'] = df['connect'].astype('int')
    df['connect_order'] = df['connect_order'].astype('int')

    return df


def test_component_idx_to_name():
    assert component_idx_to_name(2) == 'valueA'
    assert component_idx_to_name(3) == 'valueB'
    assert component_idx_to_name(4) is None
    assert component_idx_to_name(1) is None
    assert component_idx_to_name(None) is None


def test_scatter_numpy():
    x = np.random.rand(500)
    y = np.random.rand(500)

    scatter = Scatter(x, y)
    widget = scatter.widget
    widget_data = np.asarray(widget.points)

    assert (500, 4) == widget_data.shape
    assert np.allclose(to_ndc(x, create_default_norm()), widget_data[:, 0])
    assert np.allclose(to_ndc(y, create_default_norm()), widget_data[:, 1])
    assert np.sum(widget_data[:, 2:]) == 0


def test_scatter_pandas(df):
    scatter = Scatter(data=df, x='a', y='b')
    widget_data = np.asarray(scatter.widget.points)

    assert (500, 4) == widget_data.shape
    assert np.allclose(to_ndc(df['a'].values, create_default_norm()), widget_data[:, 0])
    assert np.allclose(to_ndc(df['b'].values, create_default_norm()), widget_data[:, 1])


def test_scatter_pandas_update(df, df2):
    x = 'a'
    y = 'b'
    scatter = Scatter(data=df, x=x, y=y)
    assert np.allclose(
        np.array([df[x].min(), df[x].max()]), np.array(scatter.widget.x_domain)
    )
    assert np.allclose(
        np.array([df[y].min(), df[y].max()]), np.array(scatter.widget.y_domain)
    )
    assert np.allclose(
        np.array([df[x].min(), df[x].max()]), np.array(scatter.widget.x_scale_domain)
    )
    assert np.allclose(
        np.array([df[y].min(), df[y].max()]), np.array(scatter.widget.y_scale_domain)
    )

    prev_x_scale_domain = np.array(scatter.widget.x_scale_domain)
    prev_y_scale_domain = np.array(scatter.widget.y_scale_domain)

    scatter.data(df2)

    # The data domain updated by the scale domain remain unchanged as the view was not reset
    assert np.allclose(
        np.array([df2[x].min(), df2[x].max()]), np.array(scatter.widget.x_domain)
    )
    assert np.allclose(
        np.array([df2[y].min(), df2[y].max()]), np.array(scatter.widget.y_domain)
    )
    assert np.allclose(prev_x_scale_domain, np.array(scatter.widget.x_scale_domain))
    assert np.allclose(prev_y_scale_domain, np.array(scatter.widget.y_scale_domain))

    scatter.data(df)
    scatter.data(df2, reset_scales=True)

    # Now that we reset the view, both the data and scale domain updated properly
    assert np.allclose(
        np.array([df2[x].min(), df2[x].max()]), np.array(scatter.widget.x_domain)
    )
    assert np.allclose(
        np.array([df2[y].min(), df2[y].max()]), np.array(scatter.widget.y_domain)
    )
    assert np.allclose(
        np.array([df2[x].min(), df2[x].max()]), np.array(scatter.widget.x_scale_domain)
    )
    assert np.allclose(
        np.array([df2[y].min(), df2[y].max()]), np.array(scatter.widget.y_scale_domain)
    )

    assert df[x].max() != df2[x].max()
    assert df[y].max() != df2[y].max()

    assert np.allclose(
        to_ndc(df2[x].values, create_default_norm()), scatter.widget.points[:, 0]
    )
    assert np.allclose(
        to_ndc(df2[y].values, create_default_norm()), scatter.widget.points[:, 1]
    )


def test_xy_scale_shorthands(df):
    for norm, Norm in [
        ('linear', Normalize),
        ('time', TimeNormalize),
        ('log', LogNorm),
        ('pow', partial(PowerNorm, gamma=2)),
    ]:
        scatter = Scatter(
            data=df,
            x='a',
            x_scale=norm,
            y='b',
            y_scale=norm,
        )
        points = np.asarray(scatter.widget.points)

        assert np.allclose(to_ndc(df['a'].values, Norm()), points[:, 0])
        assert np.allclose(to_ndc(df['b'].values, Norm()), points[:, 1])


def test_xy_manual_scale(df):
    for Norm in [Normalize, LogNorm, partial(PowerNorm, gamma=2)]:
        scatter = Scatter(
            data=df,
            x='a',
            x_scale=Norm(),
            y='b',
            y_scale=Norm(),
        )
        points = np.asarray(scatter.widget.points)

        assert np.allclose(to_ndc(df['a'].values, Norm()), points[:, 0])
        assert np.allclose(to_ndc(df['b'].values, Norm()), points[:, 1])


def test_scatter_point_encoding_updates(df: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b')
    widget = scatter.widget
    widget_data = np.asarray(widget.points)

    assert len(scatter._encodings.data) == 0
    assert np.sum(widget_data[:, 2]) == 0

    scatter.color(by='group')
    widget_data = np.asarray(widget.points)
    assert 'color' in scatter._encodings.visual
    assert 'group:linear' in scatter._encodings.data
    assert np.sum(widget_data[:, 2]) > 0
    assert np.sum(widget_data[:, 3]) == 0

    scatter.opacity(by='c')
    widget_data = np.asarray(widget.points)
    assert 'opacity' in scatter._encodings.visual
    assert 'c:linear' in scatter._encodings.data
    assert np.sum(widget_data[:, 3]) > 0

    scatter.size(by='c')
    widget_data = np.asarray(widget.points)
    assert 'size' in scatter._encodings.visual
    assert 'c:linear' in scatter._encodings.data
    assert np.sum(widget_data[:, 3]) > 0


def test_scatter_connection_by(df: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b')
    widget = scatter.widget

    scatter.connect(by='connect')
    widget_data = np.asarray(widget.points)
    assert widget_data.shape == (500, 5)
    assert np.all(df['connect'].values == widget_data[:, 4].astype(df['connect'].dtype))


def test_scatter_connection_order(df: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b')
    widget = scatter.widget

    scatter.connect(by='connect', order='connect_order')
    widget_data = np.asarray(widget.points)
    assert widget_data.shape == (500, 6)
    assert np.all(
        df['connect_order'].values
        == widget_data[:, 5].astype(df['connect_order'].dtype)
    )


def test_missing_values_handling():
    with_nan = np.array([0, 0.25, 0.5, np.nan, 1])
    no_nan = np.array([0, 0.25, 0.5, 0.75, 1])

    df = pd.DataFrame(
        {
            'x': with_nan,
            'y': with_nan,
            'z': with_nan,
            'w': with_nan,
        }
    )

    base_warning = 'data contains missing values. Those missing values will be replaced with zeros.'

    with pytest.warns(UserWarning, match=f'X {base_warning}'):
        scatter = Scatter(data=pd.DataFrame({'x': with_nan, 'y': no_nan}), x='x', y='y')
        print(scatter.widget.points)
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 0] == -1

    with pytest.warns(UserWarning, match=f'Y {base_warning}'):
        scatter = Scatter(data=pd.DataFrame({'x': no_nan, 'y': with_nan}), x='x', y='y')
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 1] == -1

    with pytest.warns(UserWarning, match=f'Color {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({'x': no_nan, 'y': no_nan, 'z': with_nan}),
            x='x',
            y='y',
            color_by='z',
        )
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 2] == 0

    with pytest.warns(UserWarning, match=f'Opacity {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({'x': no_nan, 'y': no_nan, 'z': with_nan}),
            x='x',
            y='y',
            opacity_by='z',
        )
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 2] == 0

    with pytest.warns(UserWarning, match=f'Size {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({'x': no_nan, 'y': no_nan, 'z': with_nan}),
            x='x',
            y='y',
            size_by='z',
        )
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 2] == 0

    with pytest.warns(UserWarning, match=f'Connection color {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({'x': no_nan, 'y': no_nan, 'z': with_nan}),
            x='x',
            y='y',
            connection_color_by='z',
        )
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 2] == 0

    with pytest.warns(UserWarning, match=f'Connection opacity {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({'x': no_nan, 'y': no_nan, 'z': with_nan}),
            x='x',
            y='y',
            connection_opacity_by='z',
        )
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 2] == 0

    with pytest.warns(UserWarning, match=f'Connection size {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({'x': no_nan, 'y': no_nan, 'z': with_nan}),
            x='x',
            y='y',
            connection_size_by='z',
        )
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 2] == 0


def test_scatter_axes_labels(df: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b')

    scatter.axes(labels=True)
    assert scatter.widget.axes_labels == ['a', 'b']

    scatter.axes(labels=False)
    assert scatter.widget.axes_labels == False

    scatter.axes(labels=['axis 1', 'axis 2'])
    assert scatter.widget.axes_labels == ['axis 1', 'axis 2']


def test_scatter_transition_points(
    df: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame
):
    scatter = Scatter(data=df, x='a', y='b')

    # Default settings
    assert scatter._transition_points == True
    assert scatter._transition_points_duration == 3000

    scatter = Scatter(
        data=df, x='a', y='b', transition_points=False, transition_points_duration=500
    )
    assert scatter._transition_points == False
    assert scatter._transition_points_duration == 500

    scatter = Scatter(data=df, x='a', y='b')

    # Default widget state
    assert scatter.widget.transition_points == False
    assert scatter.widget.transition_points_duration == 3000

    # Since `scatter._transition_points` is `True` and `animate` is undefined,
    # points should be transitioned by default if the x or y channel changes
    scatter.x('c')
    assert scatter.widget.transition_points == True

    scatter.y('d')
    assert scatter.widget.transition_points == True

    scatter.xy(x='a', y='b')
    assert scatter.widget.transition_points == True

    # Even though `scatter._transition_points` is still `True` but `animate` is
    # now `False`, points should **not** be transitioned if the x or y channel
    # changes
    scatter.x('c', animate=False)
    assert scatter.widget.transition_points == False

    scatter.y('d', animate=False)
    assert scatter.widget.transition_points == False

    scatter.xy(x='a', y='b', animate=False)
    assert scatter.widget.transition_points == False

    # By changing the
    scatter.options(transition_points=False)

    # Since `scatter._transition_points` is now `False` and `animate` is still
    # undefined, points should **not** be transitioned by default if the x or y
    # channel changes
    scatter.x('c')
    assert scatter.widget.transition_points == False

    scatter.y('d')
    assert scatter.widget.transition_points == False

    scatter.xy(x='a', y='b')
    assert scatter.widget.transition_points == False

    # Even though `scatter._transition_points` is still `False` but `animate` is
    # now `True`, points should be transitioned if the x or y channel changes
    scatter.x('c', animate=True)
    assert scatter.widget.transition_points == True

    scatter.y('d', animate=True)
    assert scatter.widget.transition_points == True

    scatter.xy(x='a', y='b', animate=True)
    assert scatter.widget.transition_points == True

    scatter.options(transition_points=True)

    # When changing the data, the animated point transition has to be
    # explicitely activated. By default, even if `scatter._transition_points` is
    # `True`, points will not be animated.
    scatter.data(df2)
    assert scatter.widget.transition_points == False

    scatter.data(df, animate=True)
    assert scatter.widget.transition_points == True

    # And even if the animation is explicitely actived, it'll only work if the
    # number of points is the same.
    scatter.data(df3, animate=True)
    assert scatter.widget.transition_points == False

    scatter.options(transition_points_duration=500)
    assert scatter.widget.transition_points_duration == 500


def test_scatter_check_encoding_dtype(df: pd.DataFrame):
    check_encoding_dtype(pd.Series([1], dtype='int'))
    check_encoding_dtype(pd.Series([0.5], dtype='float'))
    check_encoding_dtype(pd.Series(['a'], dtype='string'))
    check_encoding_dtype(pd.Series(['a']))
    check_encoding_dtype(pd.Series(['a'], dtype='category'))

    scatter = Scatter(data=df, x='a', y='b', color_by='group')
    check_encoding_dtype(scatter.color_data)

    with pytest.raises(ValueError):
        check_encoding_dtype(pd.Series(np.array([1 + 0j])))


def test_tooltip(df: pd.DataFrame):
    # Test initializing a scatter plot with tooltip properties
    scatter = Scatter(
        data=df, x='a', y='b', tooltip=True, tooltip_properties=['a', 'b', 'c', 'group']
    )
    assert scatter.widget.tooltip_enable == True
    assert scatter.widget.tooltip_properties == ['a', 'b', 'c', 'group']
    assert scatter.widget.tooltip_histograms == True
    assert scatter.widget.tooltip_histograms_size == 'small'

    normalized_x_histogram = np.histogram(df['a'].values, bins=20)[0]
    normalized_x_histogram = normalized_x_histogram / normalized_x_histogram.max()
    assert np.array_equal(scatter.widget.x_histogram, normalized_x_histogram)

    normalized_y_histogram = np.histogram(df['b'].values, bins=20)[0]
    normalized_y_histogram = normalized_y_histogram / normalized_y_histogram.max()
    assert np.array_equal(scatter.widget.y_histogram, normalized_y_histogram)

    assert 'c' in scatter.widget.tooltip_properties_non_visual_info
    assert 'group' in scatter.widget.tooltip_properties_non_visual_info

    normalized_c_histogram = np.histogram(df['c'].values, bins=20)[0]
    normalized_c_histogram = normalized_c_histogram / normalized_c_histogram.max()
    assert np.array_equal(
        scatter.widget.tooltip_properties_non_visual_info['c']['histogram'],
        normalized_c_histogram,
    )

    normalized_group_histogram = (
        df['group'].copy().astype(str).astype('category').cat.codes.value_counts()
    )
    normalized_group_histogram = [
        y
        for _, y in sorted(
            dict(normalized_group_histogram / normalized_group_histogram.sum()).items()
        )
    ]
    assert np.array_equal(
        scatter.widget.tooltip_properties_non_visual_info['group']['histogram'],
        normalized_group_histogram,
    )

    # Test updating tooltip properties
    scatter.tooltip(properties=['a', 'c'])
    assert scatter.widget.tooltip_properties == ['a', 'c']

    # Test disabling tooltip
    scatter.tooltip(False)
    assert scatter.widget.tooltip_enable == False

    # Test enabling tooltip without specifying properties
    scatter = Scatter(data=df, x='b', y='d')
    scatter.tooltip(True)
    assert scatter.widget.tooltip_enable == True
    assert scatter.widget.tooltip_properties == ['x', 'y', 'color', 'opacity', 'size']

    # Test with invalid property
    scatter.tooltip(properties=['color', 'invalid_column'])
    assert scatter.widget.tooltip_properties == ['color']


def test_point_size_scale(df: pd.DataFrame):
    # Test initializing a scatter plot with tooltip properties
    scatter = Scatter(data=df, x='a', y='b')

    # Default size scale function is 'asinh'
    assert scatter.widget.size_scale_function == 'asinh'

    # Test changing the size scale function to 'linear'
    scatter.size(scale_function='linear')
    assert scatter.widget.size_scale_function == 'linear'

    # Test changing the size scale function to 'constant'
    scatter.size(scale_function='constant')
    assert scatter.widget.size_scale_function == 'constant'

    # Test initializing with the size scale function to 'constant'
    assert (
        Scatter(
            data=df, x='a', y='b', size_scale_function='linear'
        ).widget.size_scale_function
        == 'linear'
    )

    # Test initializing with the size scale function to 'constant'
    assert (
        Scatter(
            data=df, x='a', y='b', size_scale_function='constant'
        ).widget.size_scale_function
        == 'constant'
    )


def test_camera_is_fixed(df: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b')

    # Check that by default, the camera is not fixed
    assert scatter.widget.camera_is_fixed == False
    assert scatter.widget.mouse_mode == 'panZoom'

    # Test fixing the camera
    scatter.camera(is_fixed=True)
    assert scatter.widget.camera_is_fixed == True
    assert scatter.widget.mouse_mode == 'lasso'

    # Test unfixing the camera
    scatter.camera(is_fixed=False)
    assert scatter.widget.camera_is_fixed == False
    assert scatter.widget.mouse_mode == 'panZoom'

    scatter.mouse(mode='lasso')
    scatter.camera(is_fixed=True)
    scatter.camera(is_fixed=False)
    assert scatter.widget.mouse_mode == 'lasso'

    scatter_b = Scatter(data=df, x='a', y='b', camera_is_fixed=True)

    # Test initializing a Scatter with a fixed camera
    assert scatter_b.widget.camera_is_fixed == True
    assert scatter_b.widget.mouse_mode == 'lasso'


def test_toolbar_buttons(df: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b')

    default_buttons = [
        'pan_zoom',
        'lasso',
        'lasso_type',
        'lasso_brush_size',
        'divider',
        'full_screen',
        'download',
        'reset',
    ]

    # Check default toolbar buttons
    assert scatter.widget.toolbar_buttons == default_buttons

    # Test show() with custom buttons
    scatter.show(buttons=['pan_zoom', 'download'])
    assert scatter.widget.toolbar_buttons == ['pan_zoom', 'download']

    # Test show() without buttons resets to default
    scatter.show()
    assert scatter.widget.toolbar_buttons == default_buttons

    # Test show() supports 'save' button
    scatter.show(buttons=['pan_zoom', 'save', 'download'])
    assert scatter.widget.toolbar_buttons == ['pan_zoom', 'save', 'download']

    # Test show() warns on unknown buttons and filters them
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        scatter.show(buttons=['pan_zoom', 'bogus'])
        assert len(w) == 1
        assert 'bogus' in str(w[0].message)
    assert scatter.widget.toolbar_buttons == ['pan_zoom']

    # Test show() returns the widget
    result = scatter.show()
    assert result is scatter.widget


def test_order_by_column(df: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b')

    # Order by a numeric column ascending (default)
    scatter.order(by='c')
    point_order = scatter.widget.point_order
    assert point_order is not None
    assert len(point_order) == len(df)
    # Verify it's a valid permutation
    assert set(point_order) == set(range(len(df)))

    # The order should correspond to sorting by column 'c'
    sorted_idx = df['c'].sort_values(ascending=True, kind='mergesort').index
    expected = np.empty(len(sorted_idx), dtype=np.uint32)
    expected[sorted_idx] = np.arange(len(sorted_idx), dtype=np.uint32)
    assert np.array_equal(point_order, expected)


def test_order_by_column_desc(df: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b')

    scatter.order(by='c', direction='desc')
    point_order = scatter.widget.point_order
    assert point_order is not None

    sorted_idx = df['c'].sort_values(ascending=False, kind='mergesort').index
    expected = np.empty(len(sorted_idx), dtype=np.uint32)
    expected[sorted_idx] = np.arange(len(sorted_idx), dtype=np.uint32)
    assert np.array_equal(point_order, expected)


def test_order_by_custom_array(df: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b')

    custom_order = np.arange(len(df), dtype=np.uint32)[::-1].copy()
    scatter.order(by=custom_order)
    assert np.array_equal(scatter.widget.point_order, custom_order)


def test_order_reset(df: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b')

    scatter.order(by='c')
    assert scatter.widget.point_order is not None

    scatter.order(by=None)
    assert scatter.widget.point_order is None


def test_order_getter(df: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b')

    result = scatter.order()
    assert result == {'by': None, 'direction': 'asc', 'na_values': 'last'}

    scatter.order(by='c', direction='desc', na_values='first')
    result = scatter.order()
    assert result == {'by': 'c', 'direction': 'desc', 'na_values': 'first'}


def test_order_fluent_api(df: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b')

    # Should return self for chaining
    result = scatter.order(by='c')
    assert result is scatter


def test_order_with_na_values():
    df = pd.DataFrame(
        {
            'x': [0.0, 1.0, 2.0, 3.0, 4.0],
            'y': [0.0, 1.0, 2.0, 3.0, 4.0],
            'val': [3.0, np.nan, 1.0, np.nan, 2.0],
        }
    )
    scatter = Scatter(data=df, x='x', y='y')

    # na_values='last' (default): NaN points drawn on top (last)
    scatter.order(by='val', na_values='last')
    order_last = scatter.widget.point_order
    # Points with NaN (indices 1, 3) should have the highest order values
    assert order_last[1] > order_last[0]
    assert order_last[1] > order_last[2]
    assert order_last[1] > order_last[4]
    assert order_last[3] > order_last[0]
    assert order_last[3] > order_last[2]
    assert order_last[3] > order_last[4]

    # na_values='first': NaN points drawn first (behind)
    scatter.order(by='val', na_values='first')
    order_first = scatter.widget.point_order
    # Points with NaN (indices 1, 3) should have the lowest order values
    assert order_first[1] < order_first[0]
    assert order_first[1] < order_first[2]
    assert order_first[1] < order_first[4]
    assert order_first[3] < order_first[0]
    assert order_first[3] < order_first[2]
    assert order_first[3] < order_first[4]


def test_order_via_constructor(df: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b', order_by='c', order_direction='desc')
    assert scatter.widget.point_order is not None
    result = scatter.order()
    assert result['by'] == 'c'
    assert result['direction'] == 'desc'
