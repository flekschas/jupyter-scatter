from dataclasses import dataclass

from .font import Font


@dataclass
class FontFamily:
    regular: Font
    bold: Font
    italic: Font
    bold_italic: Font
