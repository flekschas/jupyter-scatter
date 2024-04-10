from enum import Enum
from typing import Tuple, Union
from typing_extensions import NotRequired, TypedDict, Literal

Auto = Literal['auto']
Rgb = Tuple[float, float, float]
Rgba = Tuple[float, float, float, float]
Color = Union[str, Rgb, Rgba]
TooltipPreviewType = Literal['text', 'image', 'audio']
TooltipPreviewImagePosition = Literal['top', 'bottom', 'left', 'right', 'center']
TooltipPreviewImageSize = Literal['contain', 'cover']

# To distinguish between None and an undefined (optional) argument, where None
# is used for unsetting and Undefined is used for skipping.
Undefined = type(
    'Undefined',
    (object,),
    { '__str__': lambda s: 'Undefined', '__repr__': lambda s: 'Undefined' }
)

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

class Auto(Enum):
    AUTO = 'auto'

class Reverse(Enum):
    REVERSE = 'reverse'

class Segment(Enum):
    SEGMENT = 'segment'

class LegendPosition(Enum):
    TOP = 'top'
    TOP_RIGHT = 'top-right'
    TOP_LEFT = 'top-left'
    BOTTOM = 'bottom'
    BOTTOM_RIGHT = 'bottom-right'
    BOTTOM_LEFT = 'bottom-left'
    RIGHT = 'right'
    LEFT = 'left'
    CENTER = 'center'

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
