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
    assert np.allclose(np.array([df[x].min(), df[x].max()]), np.array(scatter.widget.x_domain))
    assert np.allclose(np.array([df[y].min(), df[y].max()]), np.array(scatter.widget.y_domain))
    assert np.allclose(np.array([df[x].min(), df[x].max()]), np.array(scatter.widget.x_scale_domain))
    assert np.allclose(np.array([df[y].min(), df[y].max()]), np.array(scatter.widget.y_scale_domain))

    prev_x_scale_domain = np.array(scatter.widget.x_scale_domain)
    prev_y_scale_domain = np.array(scatter.widget.y_scale_domain)

    scatter.data(df2)

    # The data domain updated by the scale domain remain unchanged as the view was not reset
    assert np.allclose(np.array([df2[x].min(), df2[x].max()]), np.array(scatter.widget.x_domain))
    assert np.allclose(np.array([df2[y].min(), df2[y].max()]), np.array(scatter.widget.y_domain))
    assert np.allclose(prev_x_scale_domain, np.array(scatter.widget.x_scale_domain))
    assert np.allclose(prev_y_scale_domain, np.array(scatter.widget.y_scale_domain))

    scatter.data(df)
    scatter.data(df2, reset_scales=True)

    # Now that we reset the view, both the data and scale domain updated properly
    assert np.allclose(np.array([df2[x].min(), df2[x].max()]), np.array(scatter.widget.x_domain))
    assert np.allclose(np.array([df2[y].min(), df2[y].max()]), np.array(scatter.widget.y_domain))
    assert np.allclose(np.array([df2[x].min(), df2[x].max()]), np.array(scatter.widget.x_scale_domain))
    assert np.allclose(np.array([df2[y].min(), df2[y].max()]), np.array(scatter.widget.y_scale_domain))

    assert df[x].max() != df2[x].max()
    assert df[y].max() != df2[y].max()

    assert np.allclose(to_ndc(df2[x].values, create_default_norm()), scatter.widget.points[:, 0])
    assert np.allclose(to_ndc(df2[y].values, create_default_norm()), scatter.widget.points[:, 1])


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
    assert np.sum(widget_data[:,2]) == 0

    scatter.color(by='group')
    widget_data = np.asarray(widget.points)
    assert 'color' in scatter._encodings.visual
    assert 'group:linear' in scatter._encodings.data
    assert np.sum(widget_data[:,2]) > 0
    assert np.sum(widget_data[:,3]) == 0

    scatter.opacity(by='c')
    widget_data = np.asarray(widget.points)
    assert 'opacity' in scatter._encodings.visual
    assert 'c:linear' in scatter._encodings.data
    assert np.sum(widget_data[:,3]) > 0

    scatter.size(by='c')
    widget_data = np.asarray(widget.points)
    assert 'size' in scatter._encodings.visual
    assert 'c:linear' in scatter._encodings.data
    assert np.sum(widget_data[:,3]) > 0


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

    df = pd.DataFrame({
        'x': with_nan,
        'y': with_nan,
        'z': with_nan,
        'w': with_nan,
    })

    base_warning = 'data contains missing values. Those missing values will be replaced with zeros.'

    with pytest.warns(UserWarning, match=f'X {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({ 'x': with_nan, 'y': no_nan }),
            x='x',
            y='y'
        )
        print(scatter.widget.points)
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 0] == -1

    with pytest.warns(UserWarning, match=f'Y {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({ 'x': no_nan, 'y': with_nan }),
            x='x',
            y='y'
        )
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 1] == -1

    with pytest.warns(UserWarning, match=f'Color {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({ 'x': no_nan, 'y': no_nan, 'z': with_nan }),
            x='x',
            y='y',
            color_by='z'
        )
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 2] == 0

    with pytest.warns(UserWarning, match=f'Opacity {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({ 'x': no_nan, 'y': no_nan, 'z': with_nan }),
            x='x',
            y='y',
            opacity_by='z'
        )
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 2] == 0

    with pytest.warns(UserWarning, match=f'Size {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({ 'x': no_nan, 'y': no_nan, 'z': with_nan }),
            x='x',
            y='y',
            size_by='z'
        )
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 2] == 0

    with pytest.warns(UserWarning, match=f'Connection color {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({ 'x': no_nan, 'y': no_nan, 'z': with_nan }),
            x='x',
            y='y',
            connection_color_by='z'
        )
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 2] == 0

    with pytest.warns(UserWarning, match=f'Connection opacity {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({ 'x': no_nan, 'y': no_nan, 'z': with_nan }),
            x='x',
            y='y',
            connection_opacity_by='z'
        )
        assert np.isfinite(scatter.widget.points).all()
        assert scatter.widget.points[3, 2] == 0

    with pytest.warns(UserWarning, match=f'Connection size {base_warning}'):
        scatter = Scatter(
            data=pd.DataFrame({ 'x': no_nan, 'y': no_nan, 'z': with_nan }),
            x='x',
            y='y',
            connection_size_by='z'
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


def test_scatter_transition_points(df: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame):
    scatter = Scatter(data=df, x='a', y='b')

    # Default settings
    assert scatter._transition_points == True
    assert scatter._transition_points_duration == 3000

    scatter = Scatter(data=df, x='a', y='b', transition_points=False, transition_points_duration=500)
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
        check_encoding_dtype(pd.Series(np.array([1+0j])))
