import ipywidgets as widgets


def to_uint8(x):
  return int(max(0, min(x * 255, 255)))


def to_hex(color):
    if isinstance(color, list):
        rgb = [to_uint8(c) for c in color]
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
    else:
        return color


def with_left_label(label_text, widget, label_width: int = 128):
    container = widgets.HBox()
    label_layout = widgets.Layout(width=f'{label_width}px')
    label = widgets.HTML(label_text, layout=label_layout)
    container.children = (label, widget)

    return container
