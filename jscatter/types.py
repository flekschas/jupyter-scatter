from enum import Enum
from typing import Union, Tuple

Rgb = Tuple[float, float, float]
Rgba = Tuple[float, float, float, float]
Color = Union[str, Rgb, Rgba]

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
