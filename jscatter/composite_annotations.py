from __future__ import annotations

import inspect
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

from abc import ABCMeta, abstractmethod
from matplotlib.colors import to_rgba
from typing import List, Optional, Union

from .annotations import (
    HLine,
    VLine,
    Line,
    Rect,
    DEFAULT_LINE_COLOR,
    DEFAULT_LINE_WIDTH,
)
from .types import Color


class CompositeAnnotation(metaclass=ABCMeta):
    @abstractmethod
    def get_annotations(self, scatter) -> List[Union[HLine, VLine, Line, Rect]]:
        return []


class Contour(CompositeAnnotation):
    """
    A contour line annotation.

    Parameters
    ----------
    by : str, optional
        a string value specifying a column of categorical values for generating separate contour lines.
    line_color : tuple of floats or str, optional
        A tuple of floats or string value specifying the line color.
    line_width : int, optional
        An Integer value specifying the line width.
    line_opacity_by_level : bool, optional
        A Boolean value specifying if the line opacity should be linearly increased from the lowest to the highest level such that the highest level is fully opaque.
    kwargs : optional
        A dictionary of additional arguments for Seaborn's `kdeplot()`.

    See Also
    --------
    scatter.annotations : Draw annotations
    Seaborn's kdeplot : https://seaborn.pydata.org/generated/seaborn.kdeplot.html

    Notes
    -----
    See https://en.wikipedia.org/wiki/Contour_line for more information on what
    a contour line plot is.
    """

    def __init__(
        self,
        by: Optional[str] = None,
        line_color: Optional[Color] = None,
        line_width: Optional[int] = None,
        line_opacity_by_level: Optional[bool] = False,
        **kwargs,
    ):
        if sys.version_info < (3, 9):
            raise Exception('The contour line annotation requires at least Python v3.9')

        self.by = by
        self.line_color = line_color
        self.line_width = line_width
        self.line_opacity_by_level = line_opacity_by_level

        self.sns_kdeplot_kws = {
            key: value
            for key, value in kwargs
            if key in inspect.getfullargspec(sns.kdeplot).kwonlyargs
        }

    def get_annotations(self, scatter):
        cmap = scatter._color_map
        data = scatter._data
        x = scatter._x
        y = scatter._y
        hue = None if self.by is None else self.by

        def get_rgba(group):
            if hue is not None and hue == scatter._color_by:
                return cmap[group] or to_rgba(DEFAULT_LINE_COLOR)

            if self.line_color is None:
                return to_rgba(DEFAULT_LINE_COLOR)

            return to_rgba(self.line_color)

        def get_alpha(level):
            if self.line_opacity_by_level:
                return level / self.sns_kdeplot_kws.get('levels', 10)
            return 1

        def get_color(group, level):
            r, g, b, a = get_rgba(group)
            alpha = get_alpha(level)
            return (r, g, b, a * alpha)

        def get_width():
            if self.line_width is None:
                return DEFAULT_LINE_WIDTH

            return self.line_width

        axes = sns.kdeplot(data=data, x=x, y=y, hue=hue, **self.sns_kdeplot_kws)
        # The `;` is important! Otherwise the plot will be rendered in Jupyter
        # Notebook/Lab
        plt.close()
        lines = []

        for k, collection in enumerate(axes.collections):
            for l, path in enumerate(collection.get_paths()):
                if path.codes is None or len(path.vertices) == 0:
                    continue

                polygon_end = np.where(path.codes == path.CLOSEPOLY)[0]
                polygon_end = polygon_end[polygon_end != len(path.codes) - 1]

                subpaths = np.vsplit(path.vertices, polygon_end + 1)

                for subpath in subpaths:
                    vertices = list(zip(subpath[:, 0], subpath[:, 1]))
                    lines.append(
                        Line(
                            vertices,
                            line_color=get_color(k, l + 1),
                            line_width=get_width(),
                        )
                    )

        return lines
