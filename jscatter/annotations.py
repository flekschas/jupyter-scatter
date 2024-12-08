from __future__ import annotations

from dataclasses import dataclass
from matplotlib.colors import to_rgba
from typing import List, Tuple, Optional

from .types import Color

DEFAULT_1D_LINE_START = None
DEFAULT_1D_LINE_END = None
DEFAULT_LINE_COLOR = '#000000'
DEFAULT_LINE_WIDTH = 1


@dataclass
class HLine:
    """
    A horizontal line annotation.

    Parameters
    ----------
    y : float optional
        A float value in the data space specifying the y coordinate at which the
        horizontal line should be drawn.
    x_start : float, optional
        A float value in the data space specifying the x coordinate at which the
        horizontal line should start.
    x_end : float, optional
        A float value in the data space specifying the x coordinate at which the
        horizontal line should end.
    line_color : tuple of floats or str, optional
        A tuple of floats or string value specifying the line color.
    line_width : int, optional
        An Integer value specifying the line width.

    See Also
    --------
    scatter.annotations : Draw annotations

    Examples
    --------
    >>> HLine(0)
    HLine(y=0, x_start=None, x_end=None, line_color=(0.0, 0.0, 0.0, 1.0), line_width=1)
    """

    y: float
    x_start: Optional[float] = DEFAULT_1D_LINE_START
    x_end: Optional[float] = DEFAULT_1D_LINE_END
    line_color: Color = DEFAULT_LINE_COLOR
    line_width: int = DEFAULT_LINE_WIDTH

    def __post_init__(self):
        self.line_color = to_rgba(self.line_color)


@dataclass
class VLine:
    """
    A vertical line annotation.

    Parameters
    ----------
    x : float optional
        A float value in the data space specifying the x coordinate at which the
        vertical line should be drawn.
    y_start : float, optional
        A float value in the data space specifying the y coordinate at which the
        vertical line should start.
    y_end : float, optional
        A float value in the data space specifying the y coordinate at which the
        vertical line should end.
    line_color : tuple of floats or str, optional
        A tuple of floats or string value specifying the line color.
    line_width : int, optional
        An Integer value specifying the line width.

    See Also
    --------
    scatter.annotations : Draw annotations

    Examples
    --------
    >>> VLine(0)
    VLine(x=0, y_start=None, y_end=None, line_color=(0.0, 0.0, 0.0, 1.0), line_width=1)
    """

    x: float
    y_start: Optional[float] = DEFAULT_1D_LINE_START
    y_end: Optional[float] = DEFAULT_1D_LINE_END
    line_color: Color = DEFAULT_LINE_COLOR
    line_width: int = DEFAULT_LINE_WIDTH

    def __post_init__(self):
        self.line_color = to_rgba(self.line_color)


@dataclass
class Rect:
    """
    A rectangle annotation.

    Parameters
    ----------
    x_start : float
        A float value in the data space specifying the x coordinate at which the
        rectangle should start.
    x_end : float
        A float value in the data space specifying the x coordinate at which the
        rectangle should end.
    y_start : float
        A float value in the data space specifying the y coordinate at which the
        rectangle should start.
    y_end : float
        A float value in the data space specifying the y coordinate at which the
        rectangle should end.
    line_color : tuple of floats or str, optional
        A tuple of floats or string value specifying the line color.
    line_width : int, optional
        An Integer value specifying the line width.

    See Also
    --------
    scatter.annotations : Draw annotations

    Examples
    --------
    >>> Rect(0)
    Rect(x_start=-1, x_end=1, y_start=-1, y_end=1, line_color=(0.0, 0.0, 0.0, 1.0), line_width=1)
    """

    x_start: float
    x_end: float
    y_start: float
    y_end: float
    line_color: Color = DEFAULT_LINE_COLOR
    line_width: int = DEFAULT_LINE_WIDTH

    def __post_init__(self):
        self.line_color = to_rgba(self.line_color)


@dataclass
class Line:
    """
    A line annotation.

    Parameters
    ----------
    vertices : float
        A float value in the data space specifying the x coordinate at which the
        vertical line should be drawn.
    line_color : tuple of floats or str, optional
        A tuple of floats or string value specifying the line color.
    line_width : int, optional
        An Integer value specifying the line width.

    See Also
    --------
    scatter.annotations : Draw annotations

    Examples
    --------
    >>> Line([(-1, -1), (0, 0), (1, 1)])
    Line(vertices=[(-1, -1), (0, 0), (1, 1)], line_color=(0.0, 0.0, 0.0, 1.0), line_width=1)
    """

    vertices: List[Tuple[float]]
    line_color: Color = DEFAULT_LINE_COLOR
    line_width: int = DEFAULT_LINE_WIDTH

    def __post_init__(self):
        self.line_color = to_rgba(self.line_color)
