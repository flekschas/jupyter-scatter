"""
pycolormap_2d.colormap_2d.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the ColorMap2D base class and its instantiable children.

Original source:
- https://pypi.org/project/pycolormap-2d/
- https://github.com/spinthil/pycolormap-2d/

Modified from the original to:
1. Use standard importlib.resources instead of importlib_resources
2. Replace nptyping references with numpy.typing (per issue #4: https://github.com/spinthil/pycolormap-2d/issues/4)

All credit for the original implementation belongs to the original authors.
This is a compatibility fork to work with NumPy v2.
"""

import abc
from importlib import resources
from typing import Any, Generic, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

Number = TypeVar(
    'Number', int, float, np.float32, np.float64, pd.Float32Dtype, pd.Float64Dtype
)


class BaseColorMap2D(Generic[Number], metaclass=abc.ABCMeta):
    """Abstract class providing the basic functionality of the 2D color map.

    :param colormap_npy_loc: The location of the numpy file that contains the
        color map data.
    :type colormap_npy_loc: str
    :param range_x: The range of input x-values. Used to adapt the color map to
        un-normalized input data.
    :type range_x: Tuple[float, float]
    :param range_y: The range of input y-values. Used to adapt the color map to
        un-normalized input data.
    :type range_y: Tuple[float, float]
    """

    _cmap_data: npt.NDArray[np.uint8]
    _cmap_width: int
    _cmap_height: int
    range_x: Tuple[float, float]
    range_y: Tuple[float, float]

    def __init__(
        self,
        colormap_npy_loc: str,
        range_x: Tuple[Number, Number],
        range_y: Tuple[Number, Number],
    ) -> None:
        self._type_check_range_args(range_x, 'range_x')
        self._type_check_range_args(range_y, 'range_y')

        self.range_x = range_x
        self.range_y = range_y

        # Load color map data from resource file.
        ref = resources.files('jscatter') / colormap_npy_loc
        self._cmap_data = np.load(ref)

        self._cmap_width = self._cmap_data.shape[0]
        self._cmap_height = self._cmap_data.shape[1]

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f'{class_name}'

    def __call__(self, x: Number, y: Number) -> npt.NDArray[np.uint8]:
        return self.sample(x, y)

    @staticmethod
    def _type_check_range_args(arg_value: Any, arg_name: str) -> None:
        if type(arg_value) is not tuple:
            raise ValueError(
                f"Argument '{arg_name}' expected to be of type Tuple[Number, "
                f'Number]. Instead was {type(arg_name).__name__}.'
            )
        if len(arg_value) != 2:
            raise ValueError(
                f"Argument '{arg_name}' is expected to have two elements. "
                f'Instead had length {len(arg_value)}.'
            )
        if type(arg_value[0]) not in [
            int,
            float,
            np.float32,
            np.float64,
            pd.Float32Dtype,
            pd.Float64Dtype,
        ] or type(arg_value[1]) not in [
            int,
            float,
            np.float32,
            np.float64,
            pd.Float32Dtype,
            pd.Float64Dtype,
        ]:
            raise ValueError(
                f"Argument '{arg_name}' expected to be of type Tuple[Number, "
                f'Number]. Instead was Tuple[{type(arg_value[0]).__name__}, '
                f'{type(arg_value[1]).__name__}].'
            )

    @staticmethod
    def _clamp(v: Number, interval: Tuple[Number, Number]) -> Number:
        return min(interval[1], max(interval[0], v))

    @staticmethod
    def _linearly_scale_value(
        value: float, from_range: Tuple[float, float], to_range: Tuple[float, float]
    ) -> float:
        return (value - from_range[0]) * (to_range[1] - to_range[0]) / (
            from_range[1] - from_range[0]
        ) + to_range[0]

    def get_cmap_data(self) -> npt.NDArray[np.uint8]:
        return self._cmap_data.copy()

    def _scale_x(self, x: float) -> float:
        return self._linearly_scale_value(
            x, self.range_x, (0.0, float(self._cmap_width - 1))
        )

    def _scale_y(self, y: float) -> float:
        return self._linearly_scale_value(
            y, self.range_y, (0.0, float(self._cmap_height - 1))
        )

    def _sample(self, x: Number, y: Number) -> npt.NDArray[np.uint8]:
        image_x = int(self._clamp(round(self._scale_x(x)), (0, self._cmap_width - 1)))
        image_y = int(self._clamp(round(self._scale_y(y)), (0, self._cmap_height - 1)))

        return self._cmap_data[image_x, image_y, :]

    @abc.abstractmethod
    def sample(self, x: Number, y: Number) -> npt.NDArray[np.uint8]:
        """Get the color value at position (x, y).

        :param x: The x-coordinate (in the x_range given in the constructor or
            [0, 1] otherwise).
        :type x: int or float
        :param y: The y-coordinate (in the y_range given in the constructor or
            [0, 1] otherwise).
        :type y: int or float
        :rtype: npt.NDArray[np.uint8]
        """
        pass


class ColorMap2DBremm(BaseColorMap2D):
    """ColorMap2D using the Bremm color map.

    :param range_x: The range of input x-values. Can be used to adapt the color
        map to un-normalized input data.
    :type range_x: Tuple[float, float]
    :param range_y: The range of input y-values. Can be used to adapt the color
        map to un-normalized input data.
    :type range_y: Tuple[float, float]
    """

    def __init__(
        self,
        range_x: Tuple[float, float] = (0.0, 1.0),
        range_y: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__('pycolormap_2d/data/bremm.npy', range_x, range_y)

    def sample(self, x: Number, y: Number) -> npt.NDArray[np.uint8]:
        return super()._sample(x, y)


class ColorMap2DCubeDiagonal(BaseColorMap2D):
    """ColorMap2D using the CubeDiagonal color map.

    :param range_x: The range of input x-values. Can be used to adapt the color
        map to un-normalized input data.
    :type range_x: Tuple[float, float]
    :param range_y: The range of input y-values. Can be used to adapt the color
        map to un-normalized input data.
    :type range_y: Tuple[float, float]
    """

    def __init__(
        self,
        range_x: Tuple[float, float] = (0.0, 1.0),
        range_y: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__('pycolormap_2d/data/cubediagonal.npy', range_x, range_y)

    def sample(self, x: Number, y: Number) -> npt.NDArray[np.uint8]:
        return super()._sample(x, y)


class ColorMap2DSchumann(BaseColorMap2D):
    """ColorMap2D using the Schumann color map.

    :param range_x: The range of input x-values. Can be used to adapt the color
        map to un-normalized input data.
    :type range_x: Tuple[float, float]
    :param range_y: The range of input y-values. Can be used to adapt the color
        map to un-normalized input data.
    :type range_y: Tuple[float, float]
    """

    def __init__(
        self,
        range_x: Tuple[float, float] = (0.0, 1.0),
        range_y: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__('pycolormap_2d/data/schumann.npy', range_x, range_y)

    def sample(self, x: Number, y: Number) -> npt.NDArray[np.uint8]:
        return super()._sample(x, y)


class ColorMap2DSteiger(BaseColorMap2D):
    """ColorMap2D using the Steiger color map.

    :param range_x: The range of input x-values. Can be used to adapt the color
        map to un-normalized input data.
    :type range_x: Tuple[float, float]
    :param range_y: The range of input y-values. Can be used to adapt the color
        map to un-normalized input data.
    :type range_y: Tuple[float, float]
    """

    def __init__(
        self,
        range_x: Tuple[float, float] = (0.0, 1.0),
        range_y: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__('pycolormap_2d/data/steiger.npy', range_x, range_y)

    def sample(self, x: Number, y: Number) -> npt.NDArray[np.uint8]:
        return super()._sample(x, y)


class ColorMap2DTeuling2(BaseColorMap2D):
    """ColorMap2D using the Teuling2 color map.

    :param range_x: The range of input x-values. Can be used to adapt the color
        map to un-normalized input data.
    :type range_x: Tuple[float, float]
    :param range_y: The range of input y-values. Can be used to adapt the color
        map to un-normalized input data.
    :type range_y: Tuple[float, float]
    """

    def __init__(
        self,
        range_x: Tuple[float, float] = (0.0, 1.0),
        range_y: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__('pycolormap_2d/data/teulingfig2.npy', range_x, range_y)

    def sample(self, x: Number, y: Number) -> npt.NDArray[np.uint8]:
        return super()._sample(x, y)


class ColorMap2DZiegler(BaseColorMap2D):
    """ColorMap2D using the Ziegler color map.

    :param range_x: The range of input x-values. Can be used to adapt the color
        map to un-normalized input data.
    :type range_x: Tuple[float, float]
    :param range_y: The range of input y-values. Can be used to adapt the color
        map to un-normalized input data.
    :type range_y: Tuple[float, float]
    """

    def __init__(
        self,
        range_x: Tuple[float, float] = (0.0, 1.0),
        range_y: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__('pycolormap_2d/data/ziegler.npy', range_x, range_y)

    def sample(self, x: Number, y: Number) -> npt.NDArray[np.uint8]:
        return super()._sample(x, y)
