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
class HLine():
    y: float
    x_start: Optional[float] = DEFAULT_1D_LINE_START
    x_end: Optional[float] = DEFAULT_1D_LINE_END
    line_color: Color = DEFAULT_LINE_COLOR
    line_width: int = DEFAULT_LINE_WIDTH

    def __post_init__(self):
        self.line_color = to_rgba(self.line_color)

@dataclass
class VLine():
    x: float
    y_start: Optional[float] = DEFAULT_1D_LINE_START
    y_end: Optional[float] = DEFAULT_1D_LINE_END
    line_color: Color = DEFAULT_LINE_COLOR
    line_width: int = DEFAULT_LINE_WIDTH

    def __post_init__(self):
        self.line_color = to_rgba(self.line_color)

@dataclass
class Rect():
    x_start: float
    x_end: float
    y_start: float
    y_end: float
    line_color: Color = DEFAULT_LINE_COLOR
    line_width: int = DEFAULT_LINE_WIDTH

    def __post_init__(self):
        self.line_color = to_rgba(self.line_color)

@dataclass
class Line():
    vertices: List[Tuple[float]]
    line_color: Color = DEFAULT_LINE_COLOR
    line_width: int = DEFAULT_LINE_WIDTH

    def __post_init__(self):
        self.line_color = to_rgba(self.line_color)
