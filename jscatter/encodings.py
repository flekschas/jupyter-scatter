from functools import reduce


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


class Encodings():
    def __init__(self, total_components = 4, reserved_components = 2):
        self.data = {}
        self.visual = {}
        self.max = total_components - reserved_components
        self.components = Components(total_components, reserved_components)

    def set(self, visual_enc, data_enc):
        # Remove previous `visual_enc` encoding
        if self.is_unique(visual_enc):
            self.delete(visual_enc)

        if data_enc not in self.data:
            assert not self.components.full, f'Only {self.max} data encodings are supported'
            # The first value specifies the component
            # The second value
            self.data[data_enc] = self.components.add(data_enc)

        self.visual[visual_enc] = data_enc

    def get(self, visual_enc):
        if visual_enc in self.visual:
            return self.data[self.visual[visual_enc]]

    def delete(self, visual_enc):
        if visual_enc in self.visual:
            data_enc = self.visual[visual_enc]

            del self.visual[visual_enc]

            if sum([v == data_enc for v in self.visual.values()]) == 0:
                self.components.delete(data_enc)
                del self.data[data_enc]

    def is_unique(self, visual_enc):
        if visual_enc not in self.visual:
            return False

        return sum(
            [v == self.visual[visual_enc] for v in self.visual.values()]
        ) == 1
