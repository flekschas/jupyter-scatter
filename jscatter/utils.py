from matplotlib.colors import LogNorm, PowerNorm, Normalize
import ipywidgets as widgets
from urllib.parse import urlparse

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

def any_not(l, value = None):
    return any([x is not value for x in l])

def tolist(l):
    try:
        return l.tolist()
    except:
        return l

def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

def sorting_to_dict(sorting):
    out = dict()
    for order_idx, original_idx in enumerate(sorting):
        out[original_idx] = order_idx
    return out

def get_default_norm():
    return Normalize(clip=True)

def to_ndc(X, norm):
    return (norm(X).data * 2) - 1

def to_scale_type(norm):
    if (isinstance(norm, LogNorm)):
        return 'log_10'

    if (isinstance(norm, PowerNorm)):
        return f'pow_{norm.gamma}'

    return 'linear'
