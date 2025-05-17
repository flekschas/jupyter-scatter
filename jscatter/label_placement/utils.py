import logging
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Set,
    TypeVar,
    Union,
    cast,
)

import pandas as pd

from ..types import Auto, LogLevel

js_columns = [
    'label',
    'x',
    'y',
    'zoom_in',
    'zoom_out',
    'zoom_fade_extent',
    'font_color',
    'font_face',
    'font_style',
    'font_weight',
    'font_size',
]

js_columns_rename = {
    'zoom_in': 'zoomIn',
    'zoom_out': 'zoomOut',
    'zoom_fade_extent': 'zoomFadeExtent',
    'font_color': 'fontColor',
    'font_face': 'fontFace',
    'font_style': 'fontStyle',
    'font_weight': 'fontWeight',
    'font_size': 'fontSize',
}

js_columns_dtypes = {
    'label': 'str',
    'x': 'float32',
    'y': 'float32',
    'zoom_in': 'float32',
    'zoom_out': 'float32',
    'zoom_fade_extent': 'float32',
    'font_color': 'category',
    'font_face': 'category',
    'font_style': 'category',
    'font_weight': 'category',
    'font_size': 'category',
}


def to_js(labels: pd.DataFrame):
    return (
        labels[js_columns].astype(js_columns_dtypes).rename(columns=js_columns_rename)
    )


def deduplicate(items: Union[List, Generator]):
    """
    Deduplicate a list of values while preserving the insertion order.

    E.g.: deduplicate([1, 1, 1, 1, 2, 3,]) => [1, 2, 3]

    Note: We need to use `dict.fromkeys()` here instead of `set()` as `set()`
    does not guarantee insertion order on a language level while
    `dict.fromkeys()` does.
    """
    return list(dict.fromkeys(items))


def flatten(list_of_lists):
    for sublist in list_of_lists:
        for item in sublist:
            yield item


def is_categorical_data(data: Any):
    return pd.CategoricalDtype.is_dtype(data) or pd.api.types.is_string_dtype(data)


def create_linear_scale(domain, range_values):
    """
    Create a linear scale function similar to D3's linear scale.

    Parameters
    ----------
    domain : list of float
        Input domain [min, max]
    range_values : list of float
        Output range [min, max]

    Returns
    -------
    callable
        Scale function that maps values from domain to range
    """
    d_min = domain[0]
    d_extent = domain[1] - domain[0]
    r_min = range_values[0]
    r_extent = range_values[1] - range_values[0]

    def scale_fn(value):
        return r_min + (value - d_min) / d_extent * r_extent

    return scale_fn


def get_unique_labels(data: pd.DataFrame, by: list[str]) -> list[str]:
    labels = [data[label_type].unique() for label_type in by]
    return [str(l) for L in labels for l in L]


T = TypeVar('T')
U = TypeVar('U')


def identity(value: T) -> T:
    return value


def remove_line_breaks(text: str) -> str:
    return ' '.join(text.split())


def map_binary_list_property(
    name: str,
    label_types: list[str],
    labels: list[str],
    value: List[str],
) -> Set[str]:
    """
    Generic function to map a list to a binary dictionary

    Parameters
    ----------
    name : str
        The name of the property being assigned (for warning messages)
    label_types : list[str]
        List of label type names
    labels : list[str]
        All unique labels
    value : list[str]
        The value(s) to assign

    Returns
    -------
    Set[str]
        Mapping of label types and individual label to assigned values. E.g.:
        ```py
        {
            "city": True,  # type-level assignment
            "city:berlin": True,  # label-specific assignment
        }
        ```
    """

    # Convert to sets for faster lookup
    label_set = set([remove_line_breaks(str(label)) for label in labels])
    label_type_set = set(label_types)

    result = set()

    is_single_type = len(label_types) == 1
    single_type = label_types[0] if is_single_type else None

    for val in value:
        if ':' in val:
            label_type, label = val.split(':', 1)
            label = remove_line_breaks(label)
            if label_type in label_type_set and label in label_set:
                # Add as label-specific entry
                result.add(f'{label_type}:{label}')
            else:
                warnings.warn(
                    f"Unknown label '{label}' of type '{label_type}' in {name} configuration."
                )
        else:
            # No colon - could be a type or label depending on context
            if val in label_type_set:
                # It's a label type
                result.add(val)
            elif is_single_type and val in label_set:
                # For single type case, interpret as label
                result.add(f'{single_type}:{val}')
            else:
                warnings.warn(
                    f"Unknown label type '{val}'. Must be one of: {label_types}."
                )

    return result


def map_property(
    name: str,
    label_types: list[str],
    labels: list[str],
    value: Union[Auto, T, List[T], Mapping[str, T]],
    default_value: T,
    current_value: Optional[Dict[str, U]] = None,
    value_transform: Optional[Callable[[T], U]] = None,
) -> Dict[str, U]:
    """
    Generic function to map properties like font face, color, size, and
    zoom ranges to label types and individual labels

    Parameters
    ----------
    name : str
        The name of the property being assigned (for warning messages)
    label_types : list[str]
        List of label type names
    labels : list[str]
        All unique labels
    value : T
        The value(s) to assign
    default_values : T
        Default values to use if not specified
    current : Dict[str, T]
        The current mapped property. If None, the return value will be
        initialized

    Returns
    -------
    Dict[str, T]
        Mapping of label types and individual label to assigned values. E.g.:
        ```py
        {
            "city": default_value,  # type-level assignment
            "city:berlin": custom_value,  # label-specific assignment
        }
        ```
    """

    # Convert to sets for faster lookup
    label_set = set([remove_line_breaks(label) for label in labels])
    label_type_set = set(label_types)

    if value_transform:
        t = value_transform
    else:

        def identity(value: T) -> U:
            return cast(U, value)

        t = identity

    if current_value is None:
        result = {label_type: t(default_value) for label_type in label_types}
    else:
        result = current_value.copy()

    if value == 'auto':
        pass

    elif not isinstance(value, (list, dict)):
        # Single value for all label types
        result = {label_type: t(cast(T, value)) for label_type in label_types}

    elif isinstance(value, list):
        # List of values corresponding to label types
        if len(value) != len(label_types):
            warnings.warn(
                f"{name.capitalize()} list length {len(value)} doesn't match "
                f'number of label types {len(label_types)}. Using default {name}.'
            )
        else:
            result = {
                label_type: t(value[i]) for i, label_type in enumerate(label_types)
            }

    elif isinstance(value, dict):
        is_single_type = len(label_types) == 1
        single_type = label_types[0] if is_single_type else None

        for key, val in value.items():
            # Check if key contains a colon (type:label format)
            if ':' in key:
                label_type, label = key.split(':', 1)
                label = remove_line_breaks(label)
                if label_type in label_type_set and label in label_set:
                    # Add as label-specific entry
                    result[f'{label_type}:{label}'] = t(val)
                else:
                    warnings.warn(
                        f"Unknown label '{label}' of type '{label_type}' in {name} configuration."
                    )
            else:
                # No colon - could be a type or label depending on context
                if key in label_type_set:
                    # It's a label type
                    result[key] = t(val)
                elif is_single_type and key in label_set:
                    # For single type case, interpret as label
                    result[f'{single_type}:{key}'] = t(val)
                else:
                    warnings.warn(
                        f"Unknown label type '{key}'. Must be one of: {label_types}."
                    )

    return result


def noop(*args, **kwargs):
    """This method does nothing"""
    pass


def configure_logging(verbosity_level: LogLevel = 'warning'):
    """
    Configure logging for the label placement module.

    Parameters
    ----------
    verbosity_level : str, default='INFO'
        Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'
    """
    level = getattr(logging, verbosity_level.upper(), logging.INFO)

    # Get the module's logger
    module_logger = logging.getLogger(__name__)

    # Configure a handler if not already configured
    if not module_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        handler.setFormatter(formatter)
        module_logger.addHandler(handler)

    # Set the level for just this module's logger
    module_logger.setLevel(level)

    # Make sure the logger doesn't propagate to the root logger
    # This ensures our settings don't affect other loggers
    module_logger.propagate = False
