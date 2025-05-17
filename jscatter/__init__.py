try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version('jupyter-scatter')
except PackageNotFoundError:
    __version__ = 'uninstalled'

from .annotations import HLine, Line, Rect, VLine
from .color_maps import glasbey_dark, glasbey_light, okabe_ito
from .compose import compose, link
from .composite_annotations import CompositeAnnotation, Contour
from .font import Font, arial
from .jscatter import Scatter, plot
from .label_placement import LabelPlacement
from .pycolormap_2d import (
    BaseColorMap2D,
    ColorMap2DBremm,
    ColorMap2DCubeDiagonal,
    ColorMap2DSchumann,
    ColorMap2DSteiger,
    ColorMap2DTeuling2,
    ColorMap2DZiegler,
)
from .utils import brighten, darken, desaturate, saturate
