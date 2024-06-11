import numpy as np
from dataclasses import dataclass
from functools import reduce
from math import floor
from typing import List, Tuple, Union, Optional

def create_legend(encoding, norm, categories, labeling=None, linspace_num=5, category_order=None):
    variable = labeling.get('variable') if labeling else None
    values = []

    if categories:
        assert len(categories) <= len(encoding), 'There need as many or more encodings than categories'
        cat_by_idx = { cat_idx: cat for cat, cat_idx in categories.items() }
        idxs = (
            sorted(cat_by_idx.keys()) # category codes
            if category_order is None
            else map(categories.get, category_order)
        )
        values = [(cat_by_idx[idx], encoding[i], None) for i, idx in enumerate(idxs) if idx is not None]
    else:
        values = [
            (norm.inverse(s), encoding[floor((len(encoding) - 1) * s)], None)
            for s in np.linspace(0, 1, linspace_num)
        ]

        if labeling:
            values[0] = (*values[0][0:2], labeling.get('minValue'))
            values[-1] = (*values[-1][0:2], labeling.get('maxValue'))

    return dict(
        variable=variable,
        values=values,
        categorical=categories is not None
    )


class Component():
    def __init__(self, index, reserved = False):
        self._index = index
        self._reserved = reserved
        self._encoding = None
        self.prepared = False

    @property
    def index(self):
        return self._index

    @property
    def component(self):
        return self._index

    @property
    def reserved(self):
        return self._reserved

    @property
    def used(self):
        return self._reserved or self._encoding is not None

    @property
    def encoding(self):
        return self._encoding

    def store(self, encoding):
        self._encoding = encoding

    def clear(self):
        self._encoding = None
        self.prepared = False


class Components():
    def __init__(self, total = 4, reserved = 2):
        # When using a RGBA float texture to store points, the first two
        # components (red and green) are reserved for the x and y coordinate
        self.total = total
        self.reserved = reserved
        self._components = {
            i: Component(i, reserved=i < self.reserved) for i in range(self.total)
        }

    @property
    def size(self):
        return reduce(
            lambda acc, i: acc + int(self._components[i].used),
            self._components,
            0
        )

    @property
    def full(self):
        return self.size >= self.total

    def add(self, encoding):
        if not self.full:
            for index, component in self._components.items():
                if not component.used:
                    component.store(encoding)
                    return component

    def delete(self, encoding):
        for index, component in self._components.items():
            if component.encoding == encoding:
                component.clear()
                break


@dataclass
class VisualEncoding():
    channel: str  # Visual channel. I.e., color, opacity, or size
    dimension: str  # Data dimension I.e., f'{column_name}_{norm}'
    legend: Optional[List[Tuple[float, Union[float, int, str]]]] = None


class Encodings():
    def __init__(self, total_components = 4, reserved_components = 2):
        self.data = {}
        self.visual = {}
        self.max = total_components - reserved_components
        self.components = Components(total_components, reserved_components)

    def set(self, channel: str, dimension: str):
        # Remove previous `channel` encoding
        if self.is_unique(channel):
            self.delete(channel)

        if dimension not in self.data:
            assert not self.components.full, f'Only {self.max} data encodings are supported'
            # The first value specifies the component
            # The second value
            self.data[dimension] = self.components.add(dimension)

        self.visual[channel] = VisualEncoding(channel, dimension)

    def get(self, channel):
        if channel in self.visual:
            return self.data[self.visual[channel].dimension]

    def get_legend(self, channel):
        if channel in self.visual:
            return self.visual[channel].legend

    def set_legend(
        self,
        channel,
        encoding,
        norm,
        categories,
        labeling = None,
        linspace_num = 5,
        category_order = None,
    ):
        if channel in self.visual:
            self.visual[channel].legend = create_legend(
                encoding,
                norm,
                categories,
                labeling,
                linspace_num,
                category_order,
            )

    def delete(self, channel):
        if channel in self.visual:
            dimension = self.visual[channel].dimension

            del self.visual[channel]

            if sum([v == dimension for v in self.visual.values()]) == 0:
                self.components.delete(dimension)
                del self.data[dimension]

    def is_unique(self, channel):
        if channel not in self.visual:
            return False

        return sum(
            [v.dimension == self.visual[channel].dimension for v in self.visual.values()]
        ) == 1
