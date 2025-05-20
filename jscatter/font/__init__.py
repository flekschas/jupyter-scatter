from .font import Font
from .font_family import FontFamily

arial = FontFamily(
    regular=Font('arial.json', 'arial', 'normal', 'normal'),
    bold=Font('arial-bold.json', 'arial', 'bold', 'normal'),
    italic=Font('arial-bold.json', 'arial', 'normal', 'italic'),
    bold_italic=Font('arial-bold.json', 'arial', 'bold', 'italic'),
)

font_name_map = {
    'arial': arial.regular,
    'regular': arial.regular,
    'arial bold': arial.bold,
    'bold': arial.bold,
    'italic': arial.italic,
    'arial italic': arial.italic,
    'arial bold italic': arial.bold_italic,
    'bold italic': arial.bold_italic,
}
