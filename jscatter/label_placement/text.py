import numpy as np

from ..font import Font


class Text:
    """
    A class for measuring text
    """

    font: Font
    font_xadvances_len: int

    def __init__(self, font: Font):
        self.font = font
        self.font_xadvances_len = len(self.font.xadvances)
        self.whitespace_width = self.font.xadvances[32]

    def valid_ord_indices(self, line):
        """
        Get valid character IDs for some text while omitting invalid IDs
        """
        for char in line:
            code = ord(char)
            if 0 <= code < self.font_xadvances_len:
                yield code

    def measure(self, text: str, font_size: float):
        """
        Measure the width and height of text
        """
        lines = [line for line in text.split('\n') if len(line) > 0]

        width = 0
        height = len(lines) * self.font.line_height

        for line in lines:
            indices = np.array(list(self.valid_ord_indices(line)))
            width = max(width, self.font.xadvances[indices].sum())

        return [width * font_size, height * font_size]

    def measure_words(self, text: str, font_size: float):
        """
        Measure the width and height of text, word by word
        """
        words = text.split()
        widths = [
            self.font.xadvances[np.array(list(self.valid_ord_indices(word)))].sum()
            * font_size
            for word in words
        ]

        return widths
