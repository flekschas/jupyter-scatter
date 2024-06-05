import numpy as np
import pandas as pd
import pytest

from functools import partial
from matplotlib.colors import AsinhNorm, LogNorm, Normalize, PowerNorm, SymLogNorm

from jscatter.jscatter import Scatter, component_idx_to_name
from jscatter.utils import create_default_norm, to_ndc, TimeNormalize


@pytest.fixture
def df() -> pd.DataFrame:
    num_groups = 8

    data = np.random.rand(500, 7)
    data[:, 2] *= 100
    data[:, 3] *= 100
    data[:, 3] = data[:, 3].astype(int)
    data[:, 4] = np.round(data[:, 4] * (num_groups - 1)).astype(int)
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
    widget = scatter.widget
    widget_data = np.asarray(widget.points)

    assert (500, 4) == np.asarray(widget.points).shape
    assert np.allclose(to_ndc(df['a'].values, create_default_norm()), widget_data[:, 0])
    assert np.allclose(to_ndc(df['b'].values, create_default_norm()), widget_data[:, 1])


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
    assert np.sum(widget_data[:, 2:]) == 0

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
