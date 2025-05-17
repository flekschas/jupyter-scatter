"""
pycolormap_2d
~~~~~~~~~~~~~

The PyColorMap 2D package, mapping 2D coordinates to colors sampled from
different 2D color maps.

Original source:
- https://pypi.org/project/pycolormap-2d/
- https://github.com/spinthil/pycolormap-2d/

Modified from the original to:
1. Use standard importlib.resources instead of importlib_resources
2. Replace nptyping references with numpy.typing (per issue #4: https://github.com/spinthil/pycolormap-2d/issues/4)

All credit for the original implementation belongs to the original authors.
This is a compatibility fork to work with NumPy v2.
"""

from .colormap_2d import (
    BaseColorMap2D,
    ColorMap2DBremm,
    ColorMap2DCubeDiagonal,
    ColorMap2DSchumann,
    ColorMap2DSteiger,
    ColorMap2DTeuling2,
    ColorMap2DZiegler,
)

__all__ = [
    'BaseColorMap2D',
    'ColorMap2DBremm',
    'ColorMap2DCubeDiagonal',
    'ColorMap2DSchumann',
    'ColorMap2DTeuling2',
    'ColorMap2DSteiger',
    'ColorMap2DZiegler',
]
