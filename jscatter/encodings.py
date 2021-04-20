class DataEncoding():
    def __init__(self, component, reserved_components = 2):
        # When using a RGBA float texture to store points, the first two
        # components (red and green) are reserved for the x and y coordinate
        self.reserved_components = reserved_components
        self._component = component
        self.prepared = False

    @property
    def component(self):
        return self._component + self.reserved_components


class Encodings():
    def __init__(self, max = 2):
        self.data = {}
        self.visual = {}
        self.max = max

    def add(self, visual_enc, data_enc):
        n = len(self.data)

        if data_enc not in self.data:
            assert n < self.max, f'Only {self.max} data encodings are supported'
            # The first value specifies the component
            # The second value
            self.data[data_enc] = DataEncoding(len(self.data))
            self.visual[visual_enc] = data_enc

    def remove(self, visual_enc):
        if visual_enc in self.visual:
            data_enc = self.visual[visual_enc]

            del self.visual[visual_enc]

            if sum([v == data_enc for v in self.visual.values()]) == 0:
                del self.data[data_enc]
