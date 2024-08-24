import numpy as np
import pandas as pd
import pytest

from jscatter.annotations import HLine, VLine, Line, Rect
from jscatter.jscatter import Scatter

@pytest.fixture
def df() -> pd.DataFrame:
    x1, y1 = np.random.normal(-1, 0.2, 1000), np.random.normal(+1, 0.05, 1000)
    x2, y2 = np.random.normal(+1, 0.2, 1000), np.random.normal(+1, 0.05, 1000)
    x3, y3 = np.random.normal(+1, 0.2, 1000), np.random.normal(-1, 0.05, 1000)
    x4, y4 = np.random.normal(-1, 0.2, 1000), np.random.normal(-1, 0.05, 1000)

    return pd.DataFrame({
        'x': np.concatenate((x1, x2, x3, x4)),
        'y': np.concatenate((y1, y2, y3, y4)),
    })

def test_hline():
    y0 = HLine(y=0, x_start=-1, x_end=1, line_color='red', line_width=2)
    assert y0.y == 0
    assert y0.x_start == -1
    assert y0.x_end == 1
    assert y0.line_color == (1.0, 0.0, 0.0, 1.0)
    assert y0.line_width == 2

def test_vline():
    x0 = VLine(x=0, y_start=-1, y_end=1, line_color='red', line_width=2)
    assert x0.x == 0
    assert x0.y_start == -1
    assert x0.y_end == 1
    assert x0.line_color == (1.0, 0.0, 0.0, 1.0)
    assert x0.line_width == 2

def test_line():
    vertices = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    l = Line(vertices=vertices, line_color='red', line_width=2)
    assert l.vertices == vertices
    assert l.line_color == (1.0, 0.0, 0.0, 1.0)
    assert l.line_width == 2

def test_rect():
    r = Rect(x_start=-1, x_end=1, y_start=-1, y_end=1, line_color='red', line_width=2)
    assert r.x_start == -1
    assert r.x_end == 1
    assert r.y_start == -1
    assert r.y_end == 1
    assert r.line_color == (1.0, 0.0, 0.0, 1.0)
    assert r.line_width == 2

def test_scatter_annotations(df: pd.DataFrame):
    x0 = VLine(0)
    y0 = HLine(0)
    c1 = Rect(x_start=-1.5, x_end=-0.5, y_start=+0.75, y_end=+1.25)
    c2 = Rect(x_start=+0.5, x_end=+1.5, y_start=+0.75, y_end=+1.25)
    c3 = Rect(x_start=+0.5, x_end=+1.5, y_start=-1.25, y_end=-0.75)
    c4 = Rect(x_start=-1.5, x_end=-0.5, y_start=-1.25, y_end=-0.75)

    annotations=[x0, y0, c1, c2, c3, c4]

    scatter = Scatter(
        data=df,
        x='x', x_scale=(-2, 2),
        y='y', y_scale=(-2, 2),
        annotations=annotations,
    )

    assert scatter.annotations()['annotations'] == annotations

    assert scatter.widget.annotations[0].x == 0
    assert scatter.widget.annotations[0].y_start == None
    assert scatter.widget.annotations[0].y_end == None

    assert scatter.widget.annotations[1].y == 0
    assert scatter.widget.annotations[1].x_start == None
    assert scatter.widget.annotations[1].x_end == None

    assert scatter.widget.annotations[2].x_start == -0.75
    assert scatter.widget.annotations[2].x_end == -0.25
    assert scatter.widget.annotations[2].y_start == 0.375
    assert scatter.widget.annotations[2].y_end == 0.625

    assert scatter.widget.annotations[3].x_start == 0.25
    assert scatter.widget.annotations[3].x_end == 0.75
    assert scatter.widget.annotations[3].y_start == 0.375
    assert scatter.widget.annotations[3].y_end == 0.625

    assert scatter.widget.annotations[4].x_start == 0.25
    assert scatter.widget.annotations[4].x_end == 0.75
    assert scatter.widget.annotations[4].y_start == -0.625
    assert scatter.widget.annotations[4].y_end == -0.375

    assert scatter.widget.annotations[5].x_start == -0.75
    assert scatter.widget.annotations[5].x_end == -0.25
    assert scatter.widget.annotations[5].y_start == -0.625
    assert scatter.widget.annotations[5].y_end == -0.375
