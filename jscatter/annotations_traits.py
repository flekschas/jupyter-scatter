from __future__ import annotations

import json

from dataclasses import asdict
from traitlets import TraitType
from typing import Any

from .annotations import (
    HLine as AHLine,
    VLine as AVLine,
    Line as ALine,
    Rect as ARect,
)

class Line(TraitType):
    info_text = 'line annotation'

    def validate(self, obj: Any, value: Any):
        if isinstance(value, ALine):
            return value
        self.error(obj, value)

class HLine(TraitType):
    info_text = 'horizontal line annotation'

    def validate(self, obj: Any, value: Any):
        if isinstance(value, AHLine):
            return value
        self.error(obj, value)

class VLine(TraitType):
    info_text = 'vertical line annotation'

    def validate(self, obj: Any, value: Any):
        if isinstance(value, AVLine):
            return value
        self.error(obj, value)

class Rect(TraitType):
    info_text = 'rectangle annotation'

    def validate(self, obj: Any, value: Any):
        if isinstance(value, ARect):
            return value
        self.error(obj, value)

def to_json(value, *args, **kwargs):
    d = None if value is None else asdict(value)
    return json.dumps(d, allow_nan=False)

def annotations_to_json(value, *args, **kwargs):
    if value is None:
        return None
    return [to_json(v) for v in value]

def from_json(value, *args, **kwargs):
    d = json.loads(value)

    if 'y' in d:
        return AHLine(**d)

    if 'x' in d:
        return AVLine(**d)

    if 'x_start' in d and 'x_end' in d and 'y_start' in d and 'y_end' in d:
        return ALine(**d)

    if 'vertices' in d:
        return ARect(**d)

    return None

def annotations_from_json(value):
    value = json.loads(value)

    if value is None:
        return None

    return [from_json(v) for v in value]

serialization = dict(
    to_json=annotations_to_json,
    from_json=annotations_from_json,
)
