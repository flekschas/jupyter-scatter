import colorsys
import math
import warnings
from typing import List, Tuple, TypeVar, Union
from urllib.parse import urlparse

import ipywidgets as widgets
import pandas as pd
from matplotlib.colors import (
    LogNorm,
    Normalize,
    PowerNorm,
    to_hex,
    to_rgb,
    to_rgba,
)
from numpy import histogram, isnan, sum

from .types import Color, Labeling, Undefined

T = TypeVar('T')


def to_uint8(x):
    return int(max(0, min(x * 255, 255)))


def calculate_luminance(color: Color):
    r, g, b = to_rgb(color)
    return math.sqrt(0.299 * r**2 + 0.587 * g**2 + 0.114 * b**2)


def brighten(color: Color, factor: float):
    """
    Brighten a color by a multiplicative factor.

    Parameters
    ----------
    color : Color
        Input color to brighten.
    factor : float
        Brightness factor. factor=2 means twice as bright.
        factor=1 means no change.

    Returns
    -------
    tuple
        Tuple of brightened (R, G, B, A) values in range [0, 1].
    """
    # Convert to RGBA
    r, g, b, a = to_rgba(color)

    # Convert RGB to HSL
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Scale the distance to white by the factor
    # This approach ensures factor=1 is no change and higher values
    # increase brightness proportionally
    l = 1.0 - ((1.0 - l) / factor)

    # Ensure lightness stays in [0, 1] range
    l = max(0.0, min(1.0, l))

    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    # Return with original alpha
    return (r, g, b, a)


def darken(color: Color, factor: float):
    """
    Darken a color by a multiplicative factor.

    Parameters
    ----------
    color : Color
        Input color to darken.
    factor : float
        Darkness factor. factor=2 means twice as dark.
        factor=1 means no change.

    Returns
    -------
    tuple
        Tuple of darkened (R, G, B, A) values in range [0, 1].
    """
    # Convert to RGBA
    r, g, b, a = to_rgba(color)

    # Convert RGB to HSL
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Scale the distance from black by dividing by the factor
    # This approach ensures factor=1 is no change and higher values
    # increase darkness proportionally
    l = l / factor

    # Ensure lightness stays in [0, 1] range
    l = max(0.0, min(1.0, l))

    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    # Return with original alpha
    return (r, g, b, a)


def saturate(color: Color, factor: float):
    """
    Increase the saturation of a color by a multiplicative factor.

    Parameters
    ----------
    color : Color
        Input color to saturate.
    factor : float
        Saturation factor. factor=2 means twice as saturated.
        factor=1 means no change.

    Returns
    -------
    tuple
        Tuple of saturated (R, G, B, A) values in range [0, 1].
    """
    # Convert to RGBA
    r, g, b, a = to_rgba(color)

    # Convert RGB to HSL
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Scale the saturation by the factor
    # This approach ensures factor=1 is no change and higher values
    # increase saturation proportionally
    s = 1.0 - ((1.0 - s) / factor)

    # Ensure saturation stays in [0, 1] range
    s = max(0.0, min(1.0, s))

    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    # Return with original alpha
    return (r, g, b, a)


def desaturate(color: Color, factor: float):
    """
    Decrease the saturation of a color by a multiplicative factor.

    Parameters
    ----------
    color : Color
        Input color to desaturate.
    factor : float
        Desaturation factor. factor=2 means twice as desaturated.
        factor=1 means no change.

    Returns
    -------
    tuple
        Tuple of desaturated (R, G, B, A) values in range [0, 1].
    """
    # Convert to RGBA
    r, g, b, a = to_rgba(color)

    # Convert RGB to HSL
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Scale the saturation by dividing by the factor
    # This approach ensures factor=1 is no change and higher values
    # increase desaturation proportionally
    s = s / factor

    # Ensure saturation stays in [0, 1] range
    s = max(0.0, min(1.0, s))

    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    # Return with original alpha
    return (r, g, b, a)


def adjust_color_for_labeling(
    color: Tuple[float, float, float, float], background_luminance: float
) -> str:
    if background_luminance > 0.5:
        return to_hex(darken(saturate(color, 2), 2))

    return to_hex(brighten(saturate(color, 2), 2))


def with_left_label(label_text, widget, label_width: int = 128):
    container = widgets.HBox()
    label_layout = widgets.Layout(width=f'{label_width}px')
    label = widgets.HTML(label_text, layout=label_layout)
    container.children = (label, widget)

    return container


def any_not(l, value=None):
    return any([x is not value for x in l])


def all_undefined(l: List[Union[Undefined, T]]) -> bool:
    return all([isinstance(x, Undefined) for x in l])


def any_undefined(l: List[Union[Undefined, T]]) -> bool:
    return any([isinstance(x, Undefined) for x in l])


def any_defined(l: List[Union[Undefined, T]]) -> bool:
    return any([not isinstance(x, Undefined) for x in l])


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


def to_scale_type(norm=None):
    if isinstance(norm, LogNorm):
        return 'log_10'

    if isinstance(norm, PowerNorm):
        return f'pow_{norm.gamma}'

    if isinstance(norm, TimeNormalize):
        return 'time'

    if isinstance(norm, Normalize):
        return 'linear'

    return 'categorical'


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


def is_categorical_data(data):
    return pd.CategoricalDtype.is_dtype(data) or pd.api.types.is_string_dtype(data)


def get_categorical_data(data):
    categorical_data = None

    if pd.CategoricalDtype.is_dtype(data):
        categorical_data = data.copy()

    elif pd.api.types.is_string_dtype(data):
        categorical_data = data.copy().astype('category')

    if categorical_data is not None and categorical_data.hasnans:
        # Create categories with your value first
        cats = ['NA'] + list(categorical_data.cat.categories)

        # Create Series with ordered categories
        categorical_data = pd.Series(data, dtype=pd.CategoricalDtype(categories=cats))
        categorical_data = categorical_data.fillna('NA')

    return categorical_data


def get_scale_type_from_df(data):
    if is_categorical_data(data):
        if data.nunique() == len(data):
            # When the number of unique values is the same as the data, then
            # the data is not nominal
            return None
        return 'categorical'

    return 'linear'


def get_domain_from_df(data):
    if is_categorical_data(data):
        # We need to recreate the categorization in case the data is just a
        # filtered view, in which case it might contain "missing" indices
        _data = get_categorical_data(data).astype(str).astype('category')
        return dict(zip(_data, _data.cat.codes))

    return [data.min(skipna=True), data.max(skipna=True)]


def get_histogram_from_df(data, bins=20, range=None):
    if is_categorical_data(data):
        categorical_data = get_categorical_data(data)
        # We need to recreate the categorization in case the data is just a
        # filtered view, in which case it might contain "missing" indices
        value_counts = (
            categorical_data.astype(str).astype('category').cat.codes.value_counts()
        )
        return [y for _, y in sorted(dict(value_counts / value_counts.sum()).items())]

    hist = histogram(data[~isnan(data)], bins=bins, range=range)

    return list(hist[0] / hist[0].max())


def get_is_valid_histogram_data(data):
    if pd.api.types.is_numeric_dtype(data):
        return True

    if pd.CategoricalDtype.is_dtype(data):
        return True

    return False


def sanitize_tooltip_properties(
    df,
    reserved_properties: List[str],
    properties: List[str],
):
    sanitized_properties = []

    for prop in properties:
        if prop in reserved_properties or (df is not None and prop in df):
            sanitized_properties.append(prop)
        else:
            continue

    return sanitized_properties


def zerofy_missing_values(values, dtype):
    if isnan(sum(values)):
        warnings.warn(
            f'{dtype} data contains missing values. Those missing values will be replaced with zeros.',
            UserWarning,
        )
        values[isnan(values)] = 0
    return values
