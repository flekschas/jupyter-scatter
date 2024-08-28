try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("jupyter-scatter")
except PackageNotFoundError:
    __version__ = "uninstalled"

from .jscatter import Scatter, plot
from .annotations import Line, HLine, VLine, Rect
from .compose import compose, link
from .color_maps import okabe_ito, glasbey_light, glasbey_dark
