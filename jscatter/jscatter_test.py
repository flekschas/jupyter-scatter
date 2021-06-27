import numpy as np
import pandas as pd

from .jscatter import Scatter, component_idx_to_name
from .utils import minmax_scale

def test_component_idx_to_name():
    assert 'valueA' == component_idx_to_name(2)
    assert 'valueB' == component_idx_to_name(3)
    assert None == component_idx_to_name(4)
    assert None == component_idx_to_name(1)
    assert None == component_idx_to_name(None)

def test_scatter_numpy():
    x = np.random.rand(500)
    y = np.random.rand(500)

    scatter = Scatter(x, y)
    widget = scatter.widget
    widget_data = np.asarray(widget.points)

    assert (500, 4) == widget_data.shape
    assert np.allclose(minmax_scale(x, (-1,1)), widget_data[:,0])
    assert np.allclose(minmax_scale(y, (-1,1)), widget_data[:,1])
    assert np.sum(widget_data[:,2:]) == 0

def get_df():
    num_groups = 8

    data = np.random.rand(500, 7)
    data[:,2] *= 100
    data[:,3] *= 100
    data[:,3] = data[:,3].astype(int)
    data[:,4] = np.round(data[:,4] * (num_groups - 1)).astype(int)
    data[:,5] = np.repeat(np.arange(100), 5).astype(int)
    data[:,6] = np.resize(np.arange(5), 500).astype(int)

    df = pd.DataFrame(
        data,
        columns=['a', 'b', 'c', 'd', 'group', 'connect', 'connect_order']
    )
    df['group'] = df['group'].astype('int').astype('category').map(lambda c: chr(65 + c), na_action=None)
    df['connect'] = df['connect'].astype('int')
    df['connect_order'] = df['connect_order'].astype('int')

    return df

def test_scatter_pandas():
    df = get_df()

    scatter = Scatter(data=df, x='a', y='b')
    widget = scatter.widget
    widget_data = np.asarray(widget.points)

    assert (500, 4) == np.asarray(widget.points).shape
    assert np.allclose(minmax_scale(df['a'].values, (-1,1)), widget_data[:,0])
    assert np.allclose(minmax_scale(df['b'].values, (-1,1)), widget_data[:,1])

def test_scatter_point_encoding_updates():
    df = get_df()

    scatter = Scatter(data=df, x='a', y='b')
    widget = scatter.widget
    widget_data = np.asarray(widget.points)

    assert len(scatter._encodings.data) == 0
    assert np.sum(widget_data[:,2:]) == 0

    scatter.color(by='group')
    widget_data = np.asarray(widget.points)
    assert 'color' in scatter._encodings.visual
    assert 'group' in scatter._encodings.data
    assert np.sum(widget_data[:,2]) > 0
    assert np.sum(widget_data[:,3]) == 0

    scatter.opacity(by='c')
    widget_data = np.asarray(widget.points)
    assert 'opacity' in scatter._encodings.visual
    assert 'c' in scatter._encodings.data
    assert np.sum(widget_data[:,3]) > 0

    scatter.size(by='c')
    widget_data = np.asarray(widget.points)
    assert 'size' in scatter._encodings.visual
    assert 'c' in scatter._encodings.data
    assert np.sum(widget_data[:,3]) > 0

def test_scatter_connection_encoding_updates():
    df = get_df()

    scatter = Scatter(data=df, x='a', y='b')
    widget = scatter.widget

    scatter.connect(by='connect')
    widget_data = np.asarray(widget.points)
    assert widget_data.shape == (500, 5)
    assert np.all(
        df['connect'].values == widget_data[:,4].astype(df['connect'].dtype)
    )

    scatter.connect(order='connect_order')
    widget_data = np.asarray(widget.points)
    assert widget_data.shape == (500, 6)
    assert np.all(
        df['connect_order'].values == widget_data[:,5].astype(df['connect_order'].dtype)
    )

