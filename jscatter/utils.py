import ipywidgets as widgets
import warnings

from matplotlib.colors import LogNorm, PowerNorm, Normalize
from numpy import histogram, isnan, sum
from pandas import CategoricalDtype, StringDtype
from urllib.parse import urlparse
from typing import List, Union

from .types import Labeling

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
    except Exception:
        return l

def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc, result.path])
    except Exception:
        return False

def sorting_to_dict(sorting):
    out = dict()
    for order_idx, original_idx in enumerate(sorting):
        out[original_idx] = order_idx
    return out

class TimeNormalize(Normalize):
    is_time = True

def create_default_norm(is_time=False):
    if is_time:
        return TimeNormalize()
    return Normalize()

def to_ndc(X, norm):
    return (norm(X).data * 2) - 1

def to_scale_type(norm = None):
    if (isinstance(norm, LogNorm)):
        return 'log_10'

    if (isinstance(norm, PowerNorm)):
        return f'pow_{norm.gamma}'

    if (isinstance(norm, TimeNormalize)):
        return 'time'

    if (isinstance(norm, Normalize)):
        return 'linear'

    return 'categorical'

def get_scale_type_from_df(data):
    if isinstance(data.dtype, CategoricalDtype) or isinstance(data.dtype, StringDtype):
        return 'categorical'

    return 'linear'

def get_domain_from_df(data):
    if isinstance(data.dtype, CategoricalDtype) or isinstance(data.dtype, StringDtype):
        # We need to recreate the categorization in case the data is just a
        # filtered view, in which case it might contain "missing" indices
        _data = data.copy().astype(str).astype('category')
        return dict(zip(_data, _data.cat.codes))

    return [data.min(), data.max()]

def create_labeling(partial_labeling, column: Union[str, None] = None) -> Labeling:
    labeling: Labeling = {}

    if isinstance(partial_labeling, dict):
        if 'minValue' in partial_labeling:
            labeling['minValue'] = partial_labeling['minValue']

        if 'maxValue' in partial_labeling:
            labeling['maxValue'] = partial_labeling['maxValue']

        if 'variable' in partial_labeling:
            labeling['variable'] = partial_labeling['variable']
    else:
        if len(partial_labeling) > 0:
            labeling['minValue'] = partial_labeling[0]

        if len(partial_labeling) > 1:
            labeling['maxValue'] = partial_labeling[1]

        if len(partial_labeling) > 2:
            labeling['variable'] = partial_labeling[2]

    if 'variable' not in labeling and column is not None:
        labeling['variable'] = column

    return labeling

def get_histogram_from_df(data, bins=20, range=None):
    if isinstance(data.dtype, CategoricalDtype) or isinstance(data.dtype, StringDtype):
        # We need to recreate the categorization in case the data is just a
        # filtered view, in which case it might contain "missing" indices
        value_counts = data.copy().astype(str).astype('category').cat.codes.value_counts()
        return [y for _, y in sorted(dict(value_counts / value_counts.sum()).items())]

    hist = histogram(data[~isnan(data)], bins=bins, range=range)

    return list(hist[0] / hist[0].max())

def sanitize_tooltip_properties(
    df,
    reserved_properties: List[str],
    properties: List[str],
):
    sanitized_properties = []

    for col in properties:
        if col in reserved_properties or (df is not None and col in df):
            sanitized_properties.append(col)
        else:
            continue

    return sanitized_properties

def zerofy_missing_values(values, dtype):
    if isnan(sum(values)):
        warnings.warn(
            f'{dtype} data contains missing values. Those missing values will be replaced with zeros.',
            UserWarning
        )
        values[isnan(values)] = 0
    return values
