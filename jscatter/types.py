from enum import Enum
from typing import Callable, Literal, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import NotRequired, TypedDict

Auto = Literal['auto']
Rgb = Tuple[float, float, float]
Rgba = Tuple[float, float, float, float]
Color = Union[str, Rgb, Rgba]
Position = Literal[
    'top',
    'top-right',
    'top-left',
    'bottom',
    'bottom-right',
    'bottom-left',
    'right',
    'left',
    'center',
]
TooltipPreviewType = Literal['text', 'image', 'audio']
TooltipPreviewImagePosition = Literal['top', 'bottom', 'left', 'right', 'center']
TooltipPreviewImageSize = Literal['contain', 'cover']
SizeScaleFunction = Literal['asinh', 'linear', 'constant']
LabelFont = Literal[
    'arial',
    'arial bold',
    'arial italic',
    'arial bold italic',
    'regular',
    'bold',
    'italic',
    'bold italic',
]
LabelScaleFunction = Literal['asinh', 'constant']
LabelPositioning = Literal['center_of_mass', 'highest_density', 'largest_cluster']
WidgetButtons = Literal[
    'pan_zoom', 'lasso', 'full_screen', 'screenshot', 'download', 'reset', 'divider'
]
LogLevel = Literal['debug', 'info', 'warning', 'error', 'critical']

NumericType = TypeVar('NumericType', int, float, np.number)

AggregationMethodLiteral = Literal['min', 'mean', 'median', 'max', 'sum']
AggregationMethodCallable = Callable[[npt.NDArray[np.float64]], NumericType]
AggregationMethod = Union[AggregationMethodLiteral, AggregationMethodCallable]

# To distinguish between None and an undefined (optional) argument, where None
# is used for unsetting and Undefined is used for skipping.
Undefined = type(
    'Undefined',
    (object,),
    {'__str__': lambda s: 'Undefined', '__repr__': lambda s: 'Undefined'},
)

# An "undefined" value
UNDEF = Undefined()


class Scales(Enum):
    LINEAR = 'linear'
    LOG = 'log'
    POW = 'pow'


class Size(Enum):
    S = 'small'
    M = 'medium'
    L = 'large'


class MouseModes(Enum):
    PAN_ZOOM = 'panZoom'
    LASSO = 'lasso'
    ROTATE = 'rotate'


class Reverse(Enum):
    REVERSE = 'reverse'


class Segment(Enum):
    SEGMENT = 'segment'


class VisualProperty(Enum):
    X = 'x'
    Y = 'y'
    COLOR = 'color'
    OPACITY = 'opacity'
    SIZE = 'size'


class Labeling(TypedDict):
    variable: NotRequired[str]
    minValue: NotRequired[str]
    maxValue: NotRequired[str]
