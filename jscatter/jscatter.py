from __future__ import annotations
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import to_rgba, Normalize, LogNorm, PowerNorm, LinearSegmentedColormap, ListedColormap
from typing import Dict, Optional, Union, List, Tuple

from .annotations import Line, HLine, VLine, Rect
from .encodings import Encodings
from .widget import JupyterScatter, SELECTION_DTYPE
from .color_maps import okabe_ito, glasbey_light, glasbey_dark
from .utils import any_not, to_ndc, tolist, uri_validator, to_scale_type, get_scale_type_from_df, get_domain_from_df, create_default_norm, create_labeling, get_histogram_from_df, sanitize_tooltip_properties, zerofy_missing_values
from .types import Auto, Color, Scales, MouseModes, Auto, Reverse, Segment, Size, LegendPosition, VisualProperty, Labeling, TooltipPreviewType, TooltipPreviewImagePosition, TooltipPreviewImageSize, Undefined

COMPONENT_CONNECT = 4
COMPONENT_CONNECT_ORDER = 5
VALID_ENCODING_TYPES = [
    pd.api.types.is_float_dtype,
    pd.api.types.is_integer_dtype,
    pd.api.types.is_string_dtype,
    pd.CategoricalDtype.is_dtype,
]
DEFAULT_HISTOGRAM_BINS = 20
DEFAULT_TRANSITION_POINTS_DURATION = 3000

# An "undefined" value
UNDEF = Undefined()

default_background_color = 'white'

visual_properties = [
    'x',
    'y',
    'color',
    'opacity',
    'size',
]

def get_non_visual_properties(properties):
    return [c for c in properties if c not in visual_properties]

def check_encoding_dtype(series):
    if not any([check(series.dtype) for check in VALID_ENCODING_TYPES]):
        raise ValueError(f'{series.name} is of an unsupported data type: {series.dtype}. Must be one of float*, int*, category, or string.')

def is_categorical_data(data):
    return pd.CategoricalDtype.is_dtype(data) or pd.api.types.is_string_dtype(data)

def get_categorical_data(data):
    categorical_data = None

    if pd.CategoricalDtype.is_dtype(data):
        categorical_data = data
    elif pd.api.types.is_string_dtype(data):
        categorical_data = data.copy().astype('category')

    return categorical_data

def component_idx_to_name(idx):
    if idx == 2:
        return 'valueA'

    if idx == 3:
        return 'valueB'

    return None

def order_map(map, order):
    ordered_map = map

    try:
        full_order = order + list(range(len(order), len(map)))
        ordered_map = [ordered_map[full_order[i]] for i, _ in enumerate(ordered_map)]
    except TypeError:
        pass

    return ordered_map[::(1 + (-2 * (order == 'reverse')))]

def get_map_order(map, categories):
    map_keys = list(map.keys())

    order = list(range(len(map)))

    for cat, code in categories.items():
        order[code] = map_keys.index(cat)

    return order

def get_scale(scatter: Scatter, channel: str):
    return to_scale_type(
        getattr(scatter, f'_{channel}_norm')
        if getattr(scatter, f'_{channel}_categories') is None
        else None
    )

def get_domain(scatter: Scatter, channel: str):
    if getattr(scatter, f'_{channel}_categories') is None:
        return (
            getattr(scatter, f'_{channel}_norm').vmin,
            getattr(scatter, f'_{channel}_norm').vmax
        )
    return getattr(scatter, f'_{channel}_categories')

def get_histogram_bins(bins: Union[int, Dict[str, int]], property: str):
    if isinstance(bins, dict):
        return bins.get(property, DEFAULT_HISTOGRAM_BINS)

    if isinstance(bins, int):
        return bins

    return DEFAULT_HISTOGRAM_BINS

def get_histogram_range(ranges, property):
    if isinstance(ranges, dict):
        return ranges.get(property)

    if isinstance(ranges, tuple):
        return ranges

    return None

def normalize_annotation(annotation, x_scale, y_scale):
    def to_ndc(value, norm):
        return (norm(value) * 2) - 1

    if isinstance(annotation, HLine):
        return HLine(
            y=to_ndc(annotation.y, y_scale),
            x_start=None if annotation.x_start is None else to_ndc(annotation.x_start, x_scale),
            x_end=None if annotation.x_end is None else to_ndc(annotation.x_end, x_scale),
            line_color=annotation.line_color,
            line_width=annotation.line_width,
        )

    if isinstance(annotation, VLine):
        return VLine(
            x=to_ndc(annotation.x, x_scale),
            y_start=None if annotation.y_start is None else to_ndc(annotation.y_start, y_scale),
            y_end=None if annotation.y_end is None else to_ndc(annotation.y_end, y_scale),
            line_color=annotation.line_color,
            line_width=annotation.line_width,
        )


    if isinstance(annotation, Line):
        return Line(
            vertices=[
                (to_ndc(v[0], x_scale), to_ndc(v[1], y_scale)) for v in annotation.vertices
            ],
            line_color=annotation.line_color,
            line_width=annotation.line_width,
        )

    if isinstance(annotation, Rect):
        return Rect(
            x_start=to_ndc(annotation.x_start, x_scale),
            x_end=to_ndc(annotation.x_end, x_scale),
            y_start=to_ndc(annotation.y_start, y_scale),
            y_end=to_ndc(annotation.y_end, y_scale),
            line_color=annotation.line_color,
            line_width=annotation.line_width,
        )

def normalize_annotations(annotations, x_scale, y_scale):
    if annotations is None:
        return None
    return [normalize_annotation(a, x_scale, y_scale) for a in annotations]



class Scatter():
    def __init__(
        self,
        x: Union[str, List[float], np.ndarray],
        y: Union[str, List[float], np.ndarray],
        data: Optional[pd.DataFrame] = None,
        **kwargs
    ):
        """
        Create a scatter instance.

        Parameters
        ----------
        x : str, array_like
            The x coordinates given as either an array-like list of coordinates
            or a string referencing a column in `data`.
        y : str, array_like
            The y coordinates given as either an array-like list of coordinates
            or a string referencing a column in `data`.
        data : pd.DataFrame, optional
            The data frame that holds the x and y coordinates as well as other
            possible dimensions that can be used for color, size, or opacity
            encoding.
        kwargs : optional
            Options to customize the scatter instance and the visual encoding.
            See https://jupyter-scatter.dev/api#properties
            for a complete list of all properties.

        Returns
        -------
        self
            The scatter instance

        See Also
        --------
        x : Set or get the x coordinates.
        y : Set or get the y coordinates.
        show : Show the scatter plot as a widget in Jupyter Lab or Notebook.
        plot : Create and immediately show a scatter plot widget

        Examples
        --------
        >>> Scatter(x=np.arange(5), y=np.arange(5))
        <jscatter.jscatter.Scatter>

        >>> scatter.x(data=df, x='weight', y='speed', color_by='length')
        <jscatter.jscatter.Scatter>
        """
        self._data = data
        self._data_use_index = kwargs.get('data_use_index', False)

        try:
            self._n = len(self._data)
        except TypeError:
            self._n = np.asarray(x).size

        self._x_scale = create_default_norm()
        self._y_scale = create_default_norm()
        self._points = np.zeros((self._n, 6))
        self._widget = None
        self._pixels = None
        self._encodings = Encodings()
        self._selected_points_ids = []
        self._selected_points_idxs = np.asarray([], dtype=SELECTION_DTYPE)
        self._filtered_points_ids = None
        self._filtered_points_idxs = None

        self._transition_points = True
        self._transition_points_duration = DEFAULT_TRANSITION_POINTS_DURATION

        # We need to set the background first in order to choose sensible
        # default colors
        self._reticle_color = 'auto'
        self._background_color = to_rgba(default_background_color)
        self._background_color_luminance = 1
        self._background_image = None
        self.background(
            kwargs.get('background_color', UNDEF),
            kwargs.get('background_image', UNDEF),
        )

        self._annotations = None
        self._lasso_color = (0, 0.666666667, 1, 1)
        self._lasso_on_long_press = True
        self._lasso_initiator = False
        self._lasso_min_delay = 10
        self._lasso_min_dist = 3
        self._x_data = None
        self._x_by = None
        self._x_histogram = None
        self._y_data = None
        self._y_by = None
        self._y_histogram = None
        self._color = (0, 0, 0, 0.66) if self._background_color_luminance > 0.5 else (1, 1, 1, 0.66)
        self._color_selected = (0, 0.55, 1, 1)
        self._color_hover = (0, 0, 0, 1) if self._background_color_luminance > 0.5 else (1, 1, 1, 1)
        self._color_data = None
        self._color_by = None
        self._color_map = None
        self._color_map_order = None
        self._color_norm = create_default_norm()
        self._color_order = None
        self._color_categories = None
        self._color_labeling = None
        self._color_histogram = None
        self._opacity = 0.66
        self._opacity_unselected = 0.5
        self._opacity_data = None
        self._opacity_by = 'density'
        self._opacity_map = None
        self._opacity_map_order = None
        self._opacity_norm = create_default_norm()
        self._opacity_order = None
        self._opacity_categories = None
        self._opacity_labeling = None
        self._opacity_histogram = None
        self._size = 3
        self._size_data = None
        self._size_by = None
        self._size_map = None
        self._size_map_order = None
        self._size_norm = create_default_norm()
        self._size_order = None
        self._size_categories = None
        self._size_labeling = None
        self._size_histogram = None
        self._connect_by = None
        self._connect_by_data = None
        self._connect_order = None
        self._connect_order_data = None
        self._connection_color = (0, 0, 0, 0.1) if self._background_color_luminance > 0.5 else (1, 1, 1, 0.1)
        self._connection_color_selected = (0, 0.55, 1, 1)
        self._connection_color_hover = (0, 0, 0, 0.66) if self._background_color_luminance > 0.5 else (1, 1, 1, 0.66)
        self._connection_color_data = None
        self._connection_color_by = None
        self._connection_color_map = None
        self._connection_color_map_order = None
        self._connection_color_norm = create_default_norm()
        self._connection_color_order = None
        self._connection_color_categories = None
        self._connection_color_labeling = None
        self._connection_opacity = 0.1
        self._connection_opacity_data = None
        self._connection_opacity_by = None
        self._connection_opacity_map = None
        self._connection_opacity_map_order = None
        self._connection_opacity_norm = create_default_norm()
        self._connection_opacity_order = None
        self._connection_opacity_categories = None
        self._connection_opacity_labeling = None
        self._connection_size = 2
        self._connection_size_data = None
        self._connection_size_by = None
        self._connection_size_map = None
        self._connection_size_map_order = None
        self._connection_size_norm = create_default_norm()
        self._connection_size_order = None
        self._connection_size_categories = None
        self._connection_size_labeling = None
        self._width = 'auto'
        self._height = 240
        self._reticle = True
        self._camera_target = [0, 0]
        self._camera_distance = 1.0
        self._camera_rotation = 0.0
        self._camera_view = None
        self._mouse_mode = 'panZoom'
        self._axes = True
        self._axes_grid = False
        self._axes_labels = False
        self._legend = False
        self._legend_position = 'top-left'
        self._legend_size = 'small'
        self._tooltip = False
        self._tooltip_properties = visual_properties.copy()
        self._tooltip_properties_non_visual = None
        self._tooltip_size = 'small'
        self._tooltip_histograms = True
        self._tooltip_histograms_bins = DEFAULT_HISTOGRAM_BINS
        self._tooltip_histograms_ranges = {}
        self._tooltip_histograms_size = 'small'
        self._tooltip_preview = None
        self._tooltip_preview_type = 'text'
        self._tooltip_preview_text_lines = 3
        self._tooltip_preview_image_background_color = 'auto'
        self._tooltip_preview_image_position = 'center'
        self._tooltip_preview_image_size = 'contain'
        self._tooltip_preview_audio_length = None
        self._tooltip_preview_audio_loop = False
        self._tooltip_preview_audio_controls = True
        self._zoom_to = None
        self._zoom_to_call_idx = 0
        self._zoom_animation = 500
        self._zoom_padding = 0.33
        self._zoom_on_selection = False
        self._zoom_on_filter = False
        self._regl_scatterplot_options = {}

        self.x(x, kwargs.get('x_scale', UNDEF))
        self.y(y, kwargs.get('y_scale', UNDEF))
        self.width(kwargs.get('width', UNDEF))
        self.height(kwargs.get('height', UNDEF))
        self.selection(
            kwargs.get('selection', UNDEF),
        )
        self.color(
            kwargs.get('color', UNDEF),
            kwargs.get('color_selected', UNDEF),
            kwargs.get('color_hover', UNDEF),
            kwargs.get('color_by', UNDEF),
            kwargs.get('color_map', UNDEF),
            kwargs.get('color_norm', UNDEF),
            kwargs.get('color_order', UNDEF),
            kwargs.get('color_labeling', UNDEF),
        )
        self.opacity(
            kwargs.get('opacity', UNDEF),
            kwargs.get('opacity_unselected', UNDEF),
            kwargs.get('opacity_by', UNDEF),
            kwargs.get('opacity_map', UNDEF),
            kwargs.get('opacity_norm', UNDEF),
            kwargs.get('opacity_order', UNDEF),
            kwargs.get('opacity_labeling', UNDEF),
        )
        self.size(
            kwargs.get('size', UNDEF),
            kwargs.get('size_by', UNDEF),
            kwargs.get('size_map', UNDEF),
            kwargs.get('size_norm', UNDEF),
            kwargs.get('size_order', UNDEF),
            kwargs.get('size_labeling', UNDEF),
        )
        self.connect(
            kwargs.get('connect_by', UNDEF),
            kwargs.get('connect_order', UNDEF)
        )
        self.connection_color(
            kwargs.get('connection_color', UNDEF),
            kwargs.get('connection_color_selected', UNDEF),
            kwargs.get('connection_color_hover', UNDEF),
            kwargs.get('connection_color_by', UNDEF),
            kwargs.get('connection_color_map', UNDEF),
            kwargs.get('connection_color_norm', UNDEF),
            kwargs.get('connection_color_order', UNDEF),
            kwargs.get('connection_color_labeling', UNDEF),
        )
        self.connection_opacity(
            kwargs.get('connection_opacity', UNDEF),
            kwargs.get('connection_opacity_by', UNDEF),
            kwargs.get('connection_opacity_map', UNDEF),
            kwargs.get('connection_opacity_norm', UNDEF),
            kwargs.get('connection_opacity_order', UNDEF),
            kwargs.get('connection_opacity_labeling', UNDEF),
        )
        self.connection_size(
            kwargs.get('connection_size', UNDEF),
            kwargs.get('connection_size_by', UNDEF),
            kwargs.get('connection_size_map', UNDEF),
            kwargs.get('connection_size_order', UNDEF),
            kwargs.get('connection_size_labeling', UNDEF),
        )
        self.lasso(
            kwargs.get('lasso_color', UNDEF),
            kwargs.get('lasso_initiator', UNDEF),
            kwargs.get('lasso_min_delay', UNDEF),
            kwargs.get('lasso_min_dist', UNDEF),
            kwargs.get('lasso_on_long_press', UNDEF),
        )
        self.reticle(
            kwargs.get('reticle', UNDEF),
            kwargs.get('reticle_color', UNDEF)
        )
        self.mouse(kwargs.get('mouse_mode', UNDEF))
        self.camera(
            kwargs.get('camera_target', UNDEF),
            kwargs.get('camera_distance', UNDEF),
            kwargs.get('camera_rotation', UNDEF),
            kwargs.get('camera_view', UNDEF),
        )
        self.axes(
            kwargs.get('axes', UNDEF),
            kwargs.get('axes_grid', UNDEF),
            kwargs.get('axes_labels', UNDEF),
        )
        self.legend(
            kwargs.get('legend', UNDEF),
            kwargs.get('legend_position', UNDEF),
            kwargs.get('legend_size', UNDEF),
        )
        self.tooltip(
            kwargs.get('tooltip', UNDEF),
            kwargs.get('tooltip_properties', UNDEF),
            kwargs.get('tooltip_histograms', UNDEF),
            kwargs.get('tooltip_histograms_bins', UNDEF),
            kwargs.get('tooltip_histograms_ranges', UNDEF),
            kwargs.get('tooltip_histograms_size', UNDEF),
            kwargs.get('tooltip_preview', UNDEF),
            kwargs.get('tooltip_preview_type', UNDEF),
            kwargs.get('tooltip_preview_text_lines', UNDEF),
            kwargs.get('tooltip_preview_image_background_color', UNDEF),
            kwargs.get('tooltip_preview_image_position', UNDEF),
            kwargs.get('tooltip_preview_image_size', UNDEF),
            kwargs.get('tooltip_preview_audio_length', UNDEF),
            kwargs.get('tooltip_preview_audio_loop', UNDEF),
            kwargs.get('tooltip_preview_audio_controls', UNDEF),
            kwargs.get('tooltip_size', UNDEF),
        )
        self.zoom(
            kwargs.get('zoom_to', UNDEF),
            kwargs.get('zoom_animation', UNDEF),
            kwargs.get('zoom_padding', UNDEF),
            kwargs.get('zoom_on_selection', UNDEF),
            kwargs.get('zoom_on_filter', UNDEF),
        )
        self.annotations(
            kwargs.get('annotations', UNDEF),
        )
        self.options(
            kwargs.get('transition_points', UNDEF),
            kwargs.get('transition_points_duration', UNDEF),
            kwargs.get('options', UNDEF),
        )

    def get_point_list(self):
        connect_by = bool(self._connect_by)
        connect_order = bool(self._connect_order)

        view = self._points

        if not connect_by:
            # To avoid having to serialize unused data
            view = view[:,:4]

        if not connect_order:
            # To avoid having to serialize unused data
            view = view[:,:5]

        return view.copy()

    def data(
        self,
        data: Optional[pd.DataFrame] = None,
        use_index: Optional[Union[bool, Undefined]] = UNDEF,
        reset_scales: Optional[bool] = False,
        reset_view: Optional[bool] = False,
        animate: Optional[bool] = False,
        **kwargs
    ) -> Union[Scatter, dict]:
        """
        Set or get the referenced Pandas DataFrame

        Parameters
        ----------
        data : pd.DataFrame, optional
            The new data frame that holds the x and y coordinates as well as
            other possible dimensions that can be used for color, size, or
            opacity encoding.
        label_index : bool, optional
            If `True` and if the data frame's index is not an instance of
            `RangeIndex` then the selection and filter methods will reference
            points by the data frame's label index instead of the row index.
        reset_scales : bool, optional
            If `True`, all scales (and norms) will be reset to the extend of the
            the new data.
        reset_view : bool, optional
            If `True`, the view will be reset to the origin.
        animate : bool, optional
            If `True`, if the number of points remain the same, and if
            `reset_scales` is `False`, the points will transition smoothly. If
            `reset_view` is `True`, the view will also transition smoothly.

        Returns
        -------
        self or dict
            If no parameter was provided a dictionary with the current `data`
            and `use_label_index` setting is returned. Otherwise, `self` is
            returned.

        See Also
        --------
        x : Set or get the x coordinates.
        y : Set or get the y coordinates.
        xy : Set or get the x and y coordinates.

        Examples
        --------
        >>> scatter.data(df)
        <jscatter.jscatter.Scatter>

        >>> scatter.data(use_label_index=True)
        <jscatter.jscatter.Scatter>

        >>> scatter.data()
        {'data': <pandas.DataFrame>, 'use_label_index': False}
        """
        if data is not None:
            old_n = self._n

            try:
                self._n = len(data)
                self._data = data

                if self._widget is not None:
                    self._widget.data = self._data
            except TypeError:
                return self

            self._points = np.zeros((self._n, 6))

            same_n = old_n == self._n

            if reset_scales == True:
                self._x_scale.vmin = None
                self._x_scale.vmax = None
                self._y_scale.vmin = None
                self._y_scale.vmax = None
                self._color_norm.vmin = None
                self._color_norm.vmax = None
                self._opacity_norm.vmin = None
                self._opacity_norm.vmax = None
                self._size_norm.vmin = None
                self._size_norm.vmax = None
                self._connection_color_norm.vmin = None
                self._connection_color_norm.vmax = None
                self._connection_opacity_norm.vmin = None
                self._connection_opacity_norm.vmax = None
                self._connection_size_norm.vmin = None
                self._connection_size_norm.vmax = None

            if 'skip_widget_update' not in kwargs:
                # We have to update the following widget props now to ensure
                # that the update propagated to the widget prior to updating the
                # points below. This seems to be a bug in Jupyter Lab but I
                # noticed that the order in which updates are registered on the
                # front-end does *not* correspond to the call order in Python!
                self.update_widget('transition_points', animate and same_n)
                self.update_widget('prevent_filter_reset', False)

            self.x(skip_widget_update=True, **self.x())
            self.y(skip_widget_update=True, **self.y())
            self.color(skip_widget_update=True, **self.color())
            self.opacity(skip_widget_update=True, **self.opacity())
            self.size(skip_widget_update=True, **self.size())
            self.connect(skip_widget_update=True, **self.connect())
            self.connection_color(skip_widget_update=True, **self.connection_color())
            self.connection_opacity(skip_widget_update=True, **self.connection_opacity())
            self.connection_size(skip_widget_update=True, **self.connection_size())

            if 'skip_widget_update' not in kwargs:
                self.update_widget('points', self.get_point_list())
                self.update_widget('x_domain', self._x_domain)
                self.update_widget('x_scale_domain', self._x_scale_domain)
                self.update_widget('y_domain', self._y_domain)
                self.update_widget('y_scale_domain', self._y_scale_domain)

            if reset_view:
                if reset_scales:
                    self.widget.reset_view()
                else:
                    animation = 3000 if animate and same_n else 0
                    self.widget.reset_view(
                        animation=animation,
                        data_extent=True,
                    )

        if use_index is not UNDEF:
            self._data_use_index = use_index

        if any_not([data, use_index], UNDEF):
            return self

        return dict(data = self._data)

    @property
    def x_data(self):
        if self._x_data is not None:
            return self._x_data
        return self._data[self._x_by]

    def x(
        self,
        x: Optional[Union[str, List[float], np.ndarray, Undefined]] = UNDEF,
        scale: Optional[Union[Scales, Tuple[float, float], LogNorm, PowerNorm, None, Undefined]] = UNDEF,
        animate: Optional[Union[bool, Undefined]] = UNDEF,
        **kwargs
    ) -> Union[Scatter, dict]:
        """
        Set or get the x coordinates.

        Parameters
        ----------
        x : str, array_like, optional
            The x coordinates given as either an array-like list of coordinates
            or a string referencing a column in `data`.
        scale : {'linear', 'time', 'log', 'pow'}, tuple of floats, matplotlib.color.LogNorm, or matplotlib.color.PowerNorm, optional
            The x scale
        animate : bool, optional
            If `True` or if `self._transition_points` is `True`, the points will
            transition smoothly.
        kwargs : optional
            Options which can be used to skip updating the widget when
            `skip_widget_update` is set to `True`

        Returns
        -------
        self or dict
            If no parameter was provided a dictionary with the current `x` and
            `scale` value is returned. Otherwise, `self` is returned.

        See Also
        --------
        y : Set or get the y coordinates.
        xy : Set or get the x and y coordinates.

        Examples
        --------
        >>> scatter.x(np.arange(5), scale='log')
        <jscatter.jscatter.Scatter>

        >>> scatter.x()
        {'x': array([0, 1, 2, 3, 4]), scale: <matplotlib.colors.LogNorm>}
        """
        if scale is not UNDEF:
            if scale is None or scale == 'linear':
                self._x_scale = create_default_norm()
            elif scale == 'time':
                self._x_scale = create_default_norm(is_time=True)
            elif scale == 'log':
                self._x_scale = LogNorm()
            elif scale == 'pow':
                self._x_scale = PowerNorm(2)
            elif isinstance(scale, LogNorm) or isinstance(scale, PowerNorm) or isinstance(scale, Normalize):
                self._x_scale = scale
            else:
                try:
                    vmin, vmax = scale
                    self._x_scale = Normalize(vmin, vmax)
                except:
                    pass

            if 'skip_widget_update' not in kwargs:
                self.update_widget('x_scale', to_scale_type(self._x_scale))

                if self._annotations is not None:
                    self.update_widget(
                        'annotations',
                        normalize_annotations(
                            self._annotations,
                            self._x_scale,
                            self._y_scale
                        )
                    )

        if x is not UNDEF:
            self._x = x
            if isinstance(x, str):
                self._x_data = None
                self._x_by = self._x
            else:
                self._x_data = pd.Series(x, index=self._data_index)
                self._x_by = 'Custom X Data'

        if x is not UNDEF or scale is not UNDEF:
            if 'skip_widget_update' not in kwargs:
                self.update_widget('prevent_filter_reset', True)
            self._points[:, 0] = zerofy_missing_values(self.x_data.values, 'X')

            self._x_min = np.min(self._points[:, 0])
            self._x_max = np.max(self._points[:, 0])
            self._x_domain = [self._x_min, self._x_max]
            self._x_histogram = get_histogram_from_df(
                self._points[:, 0],
                self.get_histogram_bins("x"),
                self.get_histogram_range("x")
            )

            # Reset scale to new domain
            if scale is UNDEF:
                try:
                    self._x_scale.vmin = self._x_min
                    self._x_scale.vmax = self._x_max
                except ValueError:
                    pass

            # Normalize x coordinate to [-1,1]
            self._points[:, 0] = to_ndc(self._points[:, 0], self._x_scale)

            self._x_scale_domain = [self._x_scale.vmin, self._x_scale.vmax]

            if 'skip_widget_update' not in kwargs:
                self.update_widget('x_title', self._x_by)
                self.update_widget('x_domain', self._x_domain)
                self.update_widget('x_scale_domain', self._x_scale_domain)
                _animate = self._transition_points if animate is UNDEF else animate
                self.update_widget('transition_points', _animate != False)
                self.update_widget('points', self.get_point_list())

        if any_not([x, scale], UNDEF):
            return self

        return dict(
            x = self._x,
            scale = self._x_scale
        )

    @property
    def y_data(self):
        if self._y_data is not None:
            return self._y_data
        return self._data[self._y_by]

    def y(
        self,
        y: Optional[Union[str, List[float], np.ndarray, Undefined]] = UNDEF,
        scale: Optional[Union[Scales, Tuple[float, float], LogNorm, PowerNorm, None, Undefined]] = UNDEF,
        animate: Optional[Union[bool, Undefined]] = UNDEF,
        **kwargs
    ) -> Union[Scatter, dict]:
        """
        Set or get the y coordinates.

        Parameters
        ----------
        y : str, array_like, optional
            The y coordinates given as either an array-like list of coordinates
            or a string referencing a column in `data`.
        scale : {'linear', 'time', 'log', 'pow'}, tuple of floats, matplotlib.color.LogNorm, or matplotlib.color.PowerNorm, optional
            The y scale
        animate : bool, optional
            If `True` or if `self._transition_points` is `True`, the points will
            transition smoothly.
        kwargs : optional
            Options which can be used to skip updating the widget when
            `skip_widget_update` is set to `True`

        Returns
        -------
        self or dict
            If no parameter was provided a dictionary with the current `y` and
            `scale` value is returned. Otherwise, `self` is returned.

        See Also
        --------
        x : Set or get the x coordinates.
        xy : Set or get the x and y coordinates.

        Examples
        --------
        >>> scatter.y(np.arange(5), scale='pow')
        <jscatter.jscatter.Scatter>

        >>> scatter.y()
        {'y': array([0, 1, 2, 3, 4]), scale: <matplotlib.colors.PowerNorm>}
        """
        if scale is not UNDEF:
            if scale is None or scale == 'linear':
                self._y_scale = create_default_norm()
            elif scale == 'time':
                self._y_scale = create_default_norm(is_time=True)
            elif scale == 'log':
                self._y_scale = LogNorm()
            elif scale == 'pow':
                self._y_scale = PowerNorm(2)
            elif isinstance(scale, LogNorm) or isinstance(scale, PowerNorm) or isinstance(scale, Normalize):
                self._y_scale = scale
            else:
                try:
                    vmin, vmax = scale
                    self._y_scale = Normalize(vmin, vmax)
                except:
                    pass

            if 'skip_widget_update' not in kwargs:
                self.update_widget('y_scale', to_scale_type(self._y_scale))

                if self._annotations is not None:
                    self.update_widget(
                        'annotations',
                        normalize_annotations(
                            self._annotations,
                            self._x_scale,
                            self._y_scale
                        )
                    )

        if y is not UNDEF:
            self._y = y
            if isinstance(y, str):
                self._y_data = None
                self._y_by = self._y
            else:
                self._y_data = pd.Series(y, index=self._data_index)
                self._y_by = 'Custom Y Data'

        if y is not UNDEF or scale is not UNDEF:
            self.update_widget('prevent_filter_reset', True)
            self._points[:, 1] = zerofy_missing_values(self.y_data.values, 'Y')

            self._y_min = np.min(self._points[:, 1])
            self._y_max = np.max(self._points[:, 1])
            self._y_domain = [self._y_min, self._y_max]
            self._y_histogram = get_histogram_from_df(
                self._points[:, 1],
                self.get_histogram_bins("y"),
                self.get_histogram_range("y")
            )

            # Reset scale to new domain
            if scale is UNDEF:
                try:
                    self._y_scale.vmin = self._y_min
                    self._y_scale.vmax = self._y_max
                except ValueError:
                    pass

            # Normalize y coordinate to [-1,1]
            self._points[:, 1] = to_ndc(self._points[:, 1], self._y_scale)

            self._y_scale_domain = [self._y_scale.vmin, self._y_scale.vmax]

            if 'skip_widget_update' not in kwargs:
                self.update_widget('y_title', self._y_by)
                self.update_widget('y_domain', self._y_domain)
                self.update_widget('y_scale_domain', self._y_scale_domain)
                _animate = self._transition_points if animate is UNDEF else animate
                self.update_widget('transition_points', _animate != False)
                self.update_widget('points', self.get_point_list())

        if any_not([y, scale], UNDEF):
            return self

        return dict(
            y = self._y,
            scale = self._y_scale
        )

    def xy(
        self,
        x: Optional[Union[str, List[float], np.ndarray, Undefined]] = UNDEF,
        y: Optional[Union[str, List[float], np.ndarray, Undefined]] = UNDEF,
        x_scale: Optional[Union[Scales, Tuple[float, float], LogNorm, PowerNorm, None, Undefined]] = UNDEF,
        y_scale: Optional[Union[Scales, Tuple[float, float], LogNorm, PowerNorm, None, Undefined]] = UNDEF,
        animate: Optional[Union[bool, Undefined]] = UNDEF,
        **kwargs
    ) -> Union[Scatter, dict]:
        """
        Set the x and y coordinates.

        When setting new coordinates, the points undergo an animated transitions
        from the old to the new coordinates.

        Parameters
        ----------
        x : str, array_like, optional
            The x coordinates given as either an array-like list of coordinates
            or a string referencing a column in `data`.
        x_scale : {'linear', 'log', 'pow'}, tuple of floats, matplotlib.color.LogNorm, or matplotlib.color.PowerNorm, optional
            The x scale
        y : str, array_like, optional
            The y coordinates given as either an array-like list of coordinates
            or a string referencing a column in `data`.
        y_scale : {'linear', 'log', 'pow'}, tuple of floats, matplotlib.color.LogNorm, or matplotlib.color.PowerNorm, optional
            The y scale
        animate : bool, optional
            If `True` or if `self._transition_points` is `True`, the points will
            transition smoothly.
        kwargs : optional
            Options which can be used to skip updating the widget when
            `skip_widget_update` is set to `True`

        Returns
        -------
        self or dict
            If no parameter was provided a dictionary with the current
            coordinates and scales is returned. Otherwise, `self` is returned.

        Notes
        -----
        This method is purely added for convenience to simplify updating both
        coordinates at the same time. This is useful for transitioning both
        coordinates at the same time.

        See Also
        --------
        x : Set or get the x coordinates.
        y : Set or get the y coordinates.

        Examples
        --------
        >>> scatter.xy(x=np.arange(5), y=np.arange(5))
        <jscatter.jscatter.Scatter>

        >>> scatter.xy(x_scale='log', y_scale='pow')
        <jscatter.jscatter.Scatter>
        """
        self.x(x, x_scale, skip_widget_update=True)
        self.y(y, y_scale, skip_widget_update=True)

        if any_not([x, y, x_scale, y_scale], UNDEF):
            if 'skip_widget_update' not in kwargs:
                self.update_widget('x_scale', to_scale_type(self._x_scale))
                self.update_widget('x_scale_domain', self._x_scale_domain)
                self.update_widget('x_domain', self._x_domain)
                self.update_widget('x_title', self._x_by)
                self.update_widget('y_scale', to_scale_type(self._y_scale))
                self.update_widget('y_scale_domain', self._y_scale_domain)
                self.update_widget('y_domain', self._y_domain)
                self.update_widget('y_title', self._y_by)
                self.update_widget('axes_labels', self.get_axes_labels())
                _animate = self._transition_points if animate is UNDEF else animate
                self.update_widget('transition_points', _animate != False)
                self.update_widget('points', self.get_point_list())

                if self._annotations is not None:
                    self.update_widget(
                        'annotations',
                        normalize_annotations(
                            self._annotations,
                            self._x_scale,
                            self._y_scale
                        )
                    )

            return self

        return dict(
            x = self._x,
            y = self._y,
            x_scale = self._x_scale,
            y_scale = self._y_scale,
        )

    def selection(
        self,
        point_ids: Optional[Union[List[int], np.ndarray, None, Undefined]] = UNDEF
    ) -> Union[Scatter, np.ndarray]:
        """
        Set or get selected points.

        Parameters
        ----------
        point_ids : array_like, optional
            The point IDs to be filtered. Depending on `data` and
            `data_use_index`, the IDs can either be the Pandas DataFrame indices
            or the points row indices. When set to `None` the selection is
            unset.

        Returns
        -------
        self or array_like
            If no parameters were provided the IDs of the currently selected \
            points are returned. Otherwise, `self` is returned.

        Examples
        --------
        >>> scatter.selection(df.query('mass < 0.5').index)')
        <jscatter.jscatter.Scatter>

        >>> scatter.selection(None)
        <jscatter.jscatter.Scatter>

        >>> scatter.selection()
        array([0, 4, 12], dtype=uint32)
        """
        if point_ids is not UNDEF:
            try:
                if self._data is not None and self._data_use_index:
                    row_idxs = self._data.index.get_indexer(point_ids)
                    self._selected_points_idxs = np.asarray(row_idxs[row_idxs >= 0], dtype=SELECTION_DTYPE)
                    self._selected_points_ids = self._data.iloc[self._selected_points_idxs].index
                else:
                    self._selected_points_idxs = np.asarray(point_ids, dtype=SELECTION_DTYPE)
                    self._selected_points_ids = self._selected_points_idxs
            except:
                if point_ids is None:
                    self._selected_points_idxs = np.asarray([], dtype=SELECTION_DTYPE)
                    self._selected_points_ids = self._selected_points_idxs
                pass

            self.update_widget('selection', self._selected_points_idxs)

            return self

        if self._widget is not None:
            row_idxs = self._widget.selection.astype(SELECTION_DTYPE)
            if self._data is not None and self._data_use_index:
                return self._data.iloc[row_idxs].index
            return row_idxs

        return self._selected_points_ids

    def filter(
        self,
        point_ids: Optional[Union[List[int], np.ndarray, None, Undefined]] = UNDEF
    ) -> Union[Scatter, np.ndarray, None]:
        """
        Set or get filtered points. When filtering down to a set of points, all
        other points will be hidden from the view.

        Parameters
        ----------
        point_ids : array_like or None, optional
            The point IDs to be filtered. Depending on `data` and
            `data_use_index`, the IDs can either be the Pandas DataFrame indices
            or the points row indices. When set to `None` the filter is unset.

        Returns
        -------
        self or array_like
            If no `point_ids` was provided the indices of the currently
            filtered points are returned. Otherwise, `self` is returned.

        Examples
        --------
        >>> scatter.filter(df.query('mass < 0.5').index)')
        <jscatter.jscatter.Scatter>

        >>> scatter.filter()
        array([0, 4, 12], dtype=uint32)
        """
        if point_ids is not UNDEF:
            try:
                if self._data is not None and self._data_use_index:
                    row_idxs = self._data.index.get_indexer(point_ids)
                    self._filtered_points_idxs = np.asarray(
                        row_idxs[row_idxs >= 0],
                        dtype=SELECTION_DTYPE
                    )
                    self._filtered_points_ids = self._data.iloc[self._filtered_points_idxs].index
                else:
                    self._filtered_points_idxs = np.asarray(point_ids, dtype=SELECTION_DTYPE)
                    self._filtered_points_ids = self._filtered_points_idxs
            except:
                if point_ids is None:
                    self._filtered_points_idxs = None
                    self._filtered_points_ids = None
                pass

            self.update_widget('filter', self._filtered_points_idxs)

            return self

        if self._widget is not None:
            row_idxs = self._widget.filter.astype(SELECTION_DTYPE)
            if self._data is not None and self._data_use_index:
                return self._data.iloc[row_idxs].index
            return row_idxs

        return self._filtered_points_ids

    @property
    def color_data(self):
        if self._color_data is not None:
            return self._color_data
        return self._data[self._color_by]

    def color(
        self,
        default: Optional[Union[Color, Auto, Undefined]] = UNDEF,
        selected: Optional[Union[Color, Undefined]] = UNDEF,
        hover: Optional[Union[Color, Auto, Undefined]] = UNDEF,
        by: Optional[Union[str, List[float], np.ndarray, Undefined]] = UNDEF,
        map: Optional[Union[Auto, str, dict, list, LinearSegmentedColormap, ListedColormap, Undefined]] = UNDEF,
        norm: Optional[Union[Tuple[float, float], Normalize, Undefined]] = UNDEF,
        order: Optional[Union[Reverse, List[int], List[str], Undefined]] = UNDEF,
        labeling: Optional[Union[Tuple[str, str], Tuple[str, str, str], Labeling, Undefined]] = UNDEF,
        **kwargs
    ) -> Union[Scatter, dict]:
        """
        Set or get the color encoding of the points.

        Parameters
        ----------
        default : matplotlib compatible color, optional
            The color to be applied uniformly to all points.
        selected : matplotlib compatible color, optional
            The color to be applied uniformly to all selected points.
        hover : matplotlib compatible color, optional
            The color to be applied uniformly to hovered points.
        by : str or array_like, optional
            The parameter is used for data-driven color encoding. It can either
            be an array-like list of values or a string referencing a column in
            the pd.DataFrame `data`.
        map : array_like, optional
            The color map used for data-driven color encoding. It can either be
            a string referencing a matplotlib color map, a matplotlib color map
            object, a list of matplotlib-compatible colors, a dictionary of
            category<->color pairs, or `auto`. When set to `auto`, jscatter will
            choose an appropriate color map.
        norm : array_like, optional
            The normalization method for data-driven color encoding. It can
            either be a tuple defining a value range that maps to `[0, 1]` with
            `matplotlib.colors.Normalize` or a matplotlib normalizer instance.
        order : array_like, optional
            The order of the color map. It can either be a list of values (for
            categorical coloring) or `reverse` to reverse the color map.
        labeling : array_like, optional
            The labeling of the minimum and maximum value as well as the data
            variable. If defined, labels are shown with the legend. The labeling
            can either be a list of strings (`[minValue, maxValue, variable]`)
            or a dictionary (`{ min: label, max: label, variable: label }`).
        kwargs : optional
            Options which can be used to skip updating the widget when
            `skip_widget_update` is set to `True`

        Returns
        -------
        self or dict
            If no parameter was provided the current color encoding settings are
            returned as a dictionary. Otherwise, `self` is returned.

        Notes
        -----
        See https://matplotlib.org/3.5.0/tutorials/colors/colors.html for valid
        matplotlib colors.
        See https://matplotlib.org/3.5.0/api/colors_api.html for valid
        matplotlib normalizer classes.

        See Also
        --------
        opacity : Set or get the opacity encoding.
        size : Set or get the size encoding.

        Examples
        --------
        >>> scatter.color('red')
        <jscatter.jscatter.Scatter>

        >>> scatter.color(by='speed', map='plasma', order='reverse')
        <jscatter.jscatter.Scatter>

        >>> scatter.color()
        {'color': (0, 0, 0, 0.66),
         'color_selected': (0, 0.55, 1, 1),
         'color_hover': (0, 0, 0, 1),
         'by': None,
         'map': None,
         'norm': <matplotlib.colors.Normalize at 0x12fa8b250>,
         'order': None,
         'labeling': None}
        """
        if default is not UNDEF:
            if default == 'auto':
                self._color = (0, 0, 0, 0.66) if self._background_color_luminance > 0.5 else (1, 1, 1, 0.66)
            else:
                try:
                    self._color = to_rgba(default)
                except ValueError:
                    pass

        if selected is not UNDEF:
            try:
                self._color_selected = to_rgba(selected)
                self.update_widget('color_selected', self._color_selected)
            except ValueError:
                pass

        if hover is not UNDEF:
            if hover == 'auto':
                self._color_hover = (0, 0, 0, 1) if self._background_color_luminance > 0.5 else (1, 1, 1, 1)
            else:
                try:
                    self._color_hover = to_rgba(hover)
                    self.update_widget('color_hover', self._color_hover)
                except ValueError:
                    pass

        if norm is not UNDEF:
            if callable(norm):
                try:
                    self._color_norm = norm
                except:
                    pass
            else:
                try:
                    vmin, vmax = norm
                    self._color_norm = Normalize(vmin, vmax)
                except:
                    if norm is None:
                        self._color_norm = create_default_norm()
                    pass

        data_updated = False
        if by is not UNDEF:
            self._color_by = by

            if by is None:
                self._encodings.delete('color')
                self._color_histogram = None

            else:
                if isinstance(by, str):
                    self._color_data = None
                else:
                    self._color_data = pd.Series(by, index=self._data_index)
                    self._color_by = 'Custom Color Data'

                self._encodings.set('color', self._color_data_dimension)

                check_encoding_dtype(self.color_data)

                categorical_data = get_categorical_data(self.color_data)
                component = self._encodings.data[self._color_data_dimension].component

                if categorical_data is not None:
                    self._color_categories = dict(zip(categorical_data, categorical_data.cat.codes))
                    self._points[:, component] = categorical_data.cat.codes
                    self._color_histogram = get_histogram_from_df(categorical_data)
                else:
                    self._color_categories = None
                    self._points[:, component] = self._color_norm(
                        zerofy_missing_values(self.color_data.values, 'Color')
                    )
                    self._color_histogram = get_histogram_from_df(
                        self._points[:, component],
                        self.get_histogram_bins("color"),
                        self.get_histogram_range("color")
                    )

                if not self._encodings.data[self._color_data_dimension].prepared:
                    data_updated = True
                    # Make sure we don't prepare the data twice
                    self._encodings.data[self._color_data_dimension].prepared = True

        elif default is not UNDEF:
            # Presumably the user wants to switch to a static color encoding
            self._color_by = None
            self._color_histogram = None
            self._encodings.delete('color')

        if order is not UNDEF:
            if order is None or order == 'reverse':
                self._color_order = order
            elif self._color_categories is not None:
                # Define order of the colors instead of changing `points[:, component_idx]`
                self._color_order = [self._color_categories[cat] for cat in order]

        if map is not UNDEF and map != 'auto' and map is not None:
            if self._color_categories is None:
                if callable(map):
                    # Assuming `map` is a Matplotlib LinearSegmentedColormap
                    self._color_map = map(range(256)).tolist()
                elif isinstance(map, str):
                    # Assuming `map` is the name of a Matplotlib LinearSegmentedColormap
                    self._color_map = plt.get_cmap(map)(range(256)).tolist()
                else:
                    # Assuming `map` is a list of colors
                    self._color_map = [to_rgba(c) for c in map]
            else:
                self._color_map_order = None
                if callable(map):
                    # Assuming `map` is a Matplotlib ListedColormap
                    self._color_map = [to_rgba(c) for c in map.colors]
                elif isinstance(map, str):
                    # Assuming `map` is the name of a Matplotlib ListedColormap
                    self._color_map = [to_rgba(c) for c in plt.get_cmap(map).colors]
                elif isinstance(map, dict):
                    # Assuming `map` is a dictionary of colors
                    self._color_map = [to_rgba(c) for c in map.values()]
                    self._color_map_order = list(map.keys())
                    self._color_order = get_map_order(map, self._color_categories)
                else:
                    # Assuming `map` is a list of colors
                    self._color_map = [to_rgba(c) for c in map]

        if (self._color_map is None or map == 'auto') and self._color_by is not None:
            # Assign default color maps
            if self._color_categories is None:
                self._color_map = plt.get_cmap('viridis')(range(256)).tolist()
            elif len(self._color_categories) > 8:
                if self._background_color_luminance < 0.5:
                    self._color_map = glasbey_light
                else:
                    self._color_map = glasbey_dark
            else:
                self._color_map = okabe_ito

        if self._color_categories is not None:
            assert len(self._color_categories) <= len(self._color_map), 'More categories than colors'

        if labeling is not UNDEF:
            if labeling is None:
                self._color_labeling = None
            else:
                column = self._color_by if isinstance(self._color_by, str) else None
                self._color_labeling = create_labeling(labeling, column)

        # Update widget and encoding domain-range
        if self._color_by is not None and self._color_map is not None:
            final_color_map = order_map(self._color_map, self._color_order)
            self.update_widget('color', final_color_map)
            self._encodings.set_legend(
                'color',
                final_color_map,
                self._color_norm,
                self._color_categories,
                self._color_labeling,
                category_order=self._color_map_order,
            )
            self.update_widget('color_scale', get_scale(self, 'color'))
            self.update_widget('color_domain', get_domain(self, 'color'))
        else:
            self.update_widget('color', self._color)

        self.update_widget('color_histogram', self._color_histogram)
        self.update_widget('color_by', self.js_color_by)
        self.update_widget('color_title', self._color_by)
        self.update_widget('legend_encoding', self.get_legend_encoding())

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('prevent_filter_reset', True)
            self.update_widget('points', self.get_point_list())

        if any_not([default, selected, hover, by, map, norm, order, labeling], UNDEF):
            return self

        return dict(
            default = self._color,
            selected = self._color_selected,
            hover = self._color_hover,
            by = self._color_by,
            map = self._color_map,
            norm = self._color_norm,
            order = self._color_order,
            labeling = self._color_labeling,
        )

    @property
    def opacity_data(self):
        if self._opacity_data is not None:
            return self._opacity_data
        return self._data[self._opacity_by]

    def opacity(
        self,
        default: Optional[Union[float, Undefined]] = UNDEF,
        unselected: Optional[Union[float, Undefined]] = UNDEF,
        by: Optional[Union[str, List[float], np.ndarray, Undefined]] = UNDEF,
        map: Optional[Union[Auto, dict, List[float], Tuple[float, float, int], Undefined]] = UNDEF,
        norm: Optional[Union[Tuple[float, float], Normalize, Undefined]] = UNDEF,
        order: Optional[Union[Reverse, List[int], List[str], Undefined]] = UNDEF,
        labeling: Optional[Union[Labeling, Undefined]] = UNDEF,
        **kwargs
    ):
        """
        Set or get the opacity encoding of the points.

        Parameters
        ----------
        default : float, optional
            The opacity to be applied uniformly to all points.
        unselected : float, optional
            The factor by which the opacity of unselected points is scaled.
            It must be in [0, 1] and is only applied if one or more points are
            selected.
        by : str or array_like, optional
            The parameter is used for data-driven opacity encoding. It can
            either be an array-like list of floats or a string referencing a
            column in the pd.DataFrame `data`.
        map : array_like, optional
            The opacity map used for data-driven opacity encoding. It can either
            be a list of floats, a dictionary of category<->opacity pairs, a
            triple specifying a `np.linspace`, or `auto`. When set to `auto`,
            jscatter will choose an appropriate opacity map.
        norm : array_like, optional
            The normalization method for data-driven opacity encoding. It can
            either a tuple defining a value range that maps to `[0, 1]` with
            `matplotlib.colors.Normalize` or a matplotlib normalizer instance.
        order : array_like, optional
            The order of the opacity map. It can either be a list of values (for
            categorical opacity values) or `reverse` to reverse the opacity map.
        labeling : array_like, optional
            The labeling of the minimum and maximum value as well as the data
            variable. If defined, labels are shown with the legend. The labeling
            can either be a list of strings (`[minValue, maxValue, variable]`)
            or a dictionary (`{ min: label, max: label, variable: label }`).
        kwargs : optional
            Options which can be used to skip updating the widget when
            `skip_widget_update` is set to `True`

        Returns
        -------
        self or dict
            If no parameter was provided the current opacity encoding settings
            are returned as a dictionary. Otherwise, `self` is returned.

        Notes
        -----
        See https://matplotlib.org/3.5.0/api/colors_api.html for valid
        matplotlib normalizer classes.
        See https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        for how the triple (start, stop, num) specifies a linear space

        See Also
        --------
        color : Set or get the color encoding.
        size : Set or get the size encoding.

        Examples
        --------
        >>> scatter.opacity(0.5)
        <jscatter.jscatter.Scatter>

        >>> scatter.opacity(by='speed', map=(0, 1, 5))
        <jscatter.jscatter.Scatter>

        >>> scatter.opacity()
        {'opacity': 0.5,
         'by': 'speed',
         'map': [0.0, 0.25, 0.5, 0.75, 1.0],
         'norm': <matplotlib.colors.Normalize at 0x12fa8b4c0>,
         'order': None,
         'labeling': None}
        """
        if default is not UNDEF:
            try:
                self._opacity = float(default)
                assert self._opacity >= 0 and self._opacity <= 1, 'Opacity must be in [0,1]'
            except ValueError:
                pass

        if unselected is not UNDEF:
            try:
                self._opacity_unselected = float(unselected)
                assert self._opacity_unselected >= 0 and self._opacity_unselected <= 1, 'Opacity scaling of unselected points must be in [0,1]'
                self.update_widget('opacity_unselected', self._opacity_unselected)
            except ValueError:
                pass

        if norm is not UNDEF:
            if callable(norm):
                try:
                    self._opacity_norm = norm
                except:
                    pass
            else:
                try:
                    vmin, vmax = norm
                    self._opacity_norm = Normalize(vmin, vmax)
                except:
                    if norm is None:
                        # Reset to default value
                        self._opacity_norm = create_default_norm()
                    pass

        data_updated = False
        if by is not UNDEF:
            self._opacity_by = by

            if by is None or by == 'density':
                self._encodings.delete('opacity')
                self._opacity_histogram = None

            else:
                if isinstance(by, str):
                    self._opacity_data = None
                else:
                    self._opacity_data = pd.Series(by, index=self._data_index)
                    self._opacity_by = 'Custom Opacity Data'

                self._encodings.set('opacity', self._opacity_data_dimension)

                check_encoding_dtype(self.opacity_data)

                component = self._encodings.data[self._opacity_data_dimension].component
                categorical_data = get_categorical_data(self.opacity_data)

                if categorical_data is not None:
                    self._points[:, component] = categorical_data.cat.codes
                    self._opacity_categories = dict(zip(categorical_data, categorical_data.cat.codes))
                    self._opacity_histogram = get_histogram_from_df(categorical_data)
                else:
                    self._points[:, component] = self._opacity_norm(
                        zerofy_missing_values(self.opacity_data.values, 'Opacity')
                    )
                    self._opacity_histogram = get_histogram_from_df(
                        self._points[:, component],
                        self.get_histogram_bins("opacity"),
                        self.get_histogram_range("opacity")
                    )
                    self._opacity_categories = None

                if not self._encodings.data[self._opacity_data_dimension].prepared:
                    data_updated = True
                    # Make sure we don't prepare the data twice
                    self._encodings.data[self._opacity_data_dimension].prepared = True

        elif default is not UNDEF:
            # Presumably the user wants to switch to a static opacity encoding
            self._opacity_by = None
            self._opacity_histogram = None
            self._encodings.delete('opacity')

        if order is not UNDEF:
            if order is None or order == 'reverse':
                self._opacity_order = order
            elif self._opacity_categories is not None:
                # Define order of the opacities instead of changing `points[:, component_idx]`
                self._opacity_order = [self._opacity_categories[cat] for cat in order]

        if map is not UNDEF and map != 'auto' and map is not None:
            self._opacity_map_order = None
            if isinstance(map, tuple):
                # Assuming `map` is a triple specifying a linear space
                self._opacity_map = np.linspace(*map)
            elif isinstance(map, dict):
                # Assuming `map` is a dictionary of opacities
                self._opacity_map = list(map.values())
                self._opacity_map_order = list(map.keys())
                self._opacity_order = get_map_order(map, self._opacity_categories)
            else:
                self._opacity_map = np.asarray(map)

        if (self._opacity_map is None or map == 'auto') and self._opacity_by is not None:
            # The best we can do is provide a linear opacity map
            if self._opacity_categories is not None:
                self._opacity_map = np.linspace(1/len(self._opacity_categories), 1, len(self._opacity_categories))
            else:
                self._opacity_map = np.linspace(1/256, 1, 256)

        self._opacity_map = tolist(self._opacity_map)

        if self._opacity_categories is not None:
            assert len(self._opacity_categories) <= len(self._opacity_map), 'More categories than opacities'

        if labeling is not UNDEF:
            if labeling is None:
                self._opacity_labeling = None
            else:
                column = self._opacity_by if isinstance(self._opacity_by, str) else None
                self._opacity_labeling = create_labeling(labeling, column)

        # Update widget
        if self._opacity_by is not None and self._opacity_map is not None:
            final_opacity_map = order_map(self._opacity_map, self._opacity_order)
            self.update_widget('opacity', final_opacity_map)
            if self._opacity_by != 'density':
                self._encodings.set_legend(
                    'opacity',
                    final_opacity_map,
                    self._opacity_norm,
                    self._opacity_categories,
                    self._opacity_labeling,
                    category_order=self._opacity_map_order,
                )
            self.update_widget('opacity_scale', get_scale(self, 'opacity'))
            self.update_widget('opacity_domain', get_domain(self, 'opacity'))
        else:
            self.update_widget('opacity', self._opacity)

        self.update_widget('opacity_histogram', self._opacity_histogram)
        self.update_widget('opacity_by', self.js_opacity_by)
        self.update_widget('opacity_title', self._opacity_by)
        self.update_widget('legend_encoding', self.get_legend_encoding())

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('prevent_filter_reset', True)
            self.update_widget('points', self.get_point_list())

        if any_not([default, unselected, by, map, norm, order, labeling], UNDEF):
            return self

        return dict(
            default = self._opacity,
            unselected = self._opacity_unselected,
            by = self._opacity_by,
            map = self._opacity_map,
            norm = self._opacity_norm,
            order = self._opacity_order,
            labeling = self._opacity_labeling,
        )

    @property
    def size_data(self):
        if self._size_data is not None:
            return self._size_data
        return self._data[self._size_by]

    def size(
        self,
        default: Optional[Union[float, Undefined]] = UNDEF,
        by: Optional[Union[str, List[float], np.ndarray, Undefined]] = UNDEF,
        map: Optional[Union[Auto, dict, List[float], Tuple[float, float, int], Undefined]] = UNDEF,
        norm: Optional[Union[Tuple[float, float], Normalize, Undefined]] = UNDEF,
        order: Optional[Union[Reverse, List[int], List[str], Undefined]] = UNDEF,
        labeling: Optional[Union[Labeling, Undefined]] = UNDEF,
        **kwargs
    ):
        """
        Set or get the size encoding of the points.

        Parameters
        ----------
        default : float, optional
            The size to be applied uniformly to all points.
        by : str or array_like, optional
            The parameter is used for data-driven size encoding. It can
            either be an array-like list of floats or a string referencing a
            column in the pd.DataFrame `data`.
        map : array_like, optional
            The size map used for data-driven size encoding. It can either
            be a list of floats, a dictionary of category<->size pairs, a
            triple specifying a `np.linspace`, or `auto`. When set to `auto`,
            jscatter will choose an appropriate size map.
        norm : array_like, optional
            The normalization method for data-driven size encoding. It can
            either a tuple defining a value range that maps to `[0, 1]` with
            `matplotlib.colors.Normalize` or a matplotlib normalizer instance.
        order : array_like, optional
            The order of the size map. It can either be a list of values (for
            categorical size values) or `reverse` to reverse the size map.
        labeling : array_like, optional
            The labeling of the minimum and maximum value as well as the data
            variable. If defined, labels are shown with the legend. The labeling
            can either be a list of strings (`[minValue, maxValue, variable]`)
            or a dictionary (`{ min: label, max: label, variable: label }`).
        kwargs : optional
            Options which can be used to skip updating the widget when
            `skip_widget_update` is set to `True`

        Returns
        -------
        self or dict
            If no parameter was provided the current size encoding settings
            are returned as a dictionary. Otherwise, `self` is returned.

        Notes
        -----
        See https://matplotlib.org/3.5.0/api/colors_api.html for valid
        matplotlib normalizer classes.
        See https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        for how the triple (start, stop, num) specifies a linear space

        See Also
        --------
        color : Set or get the color encoding.
        opacity : Set or get the opacity encoding.

        Examples
        --------
        >>> scatter.size(3)
        <jscatter.jscatter.Scatter>

        >>> scatter.size(by='weight', map=(2, 20, 4))
        <jscatter.jscatter.Scatter>

        >>> scatter.size()
        {'size': 3,
         'by': 'weight',
         'map': [2.0, 8.0, 14.0, 20.0],
         'norm': <matplotlib.colors.Normalize at 0x12fa8b580>,
         'order': None,
         'labeling': None}
        """
        if default is not UNDEF:
            try:
                self._size = int(default)
                assert self._size > 0, 'Size must be a positive integer'
                self.update_widget('size', self._size_map or self._size)
            except ValueError:
                pass

        if norm is not UNDEF:
            if callable(norm):
                try:
                    self._size_norm = norm
                except:
                    pass
            else:
                try:
                    vmin, vmax = norm
                    self._size_norm = Normalize(vmin, vmax)
                except:
                    if norm is not None:
                        self._size_norm = create_default_norm()
                    pass

        data_updated = False
        if by is not UNDEF:
            self._size_by = by

            if by is None:
                self._encodings.delete('size')
                self._size_histogram = None

            else:
                if isinstance(by, str):
                    self._size_data = None
                else:
                    self._size_data = pd.Series(by, index=self._data_index)
                    self._size_by = 'Custom Size Data'

                self._encodings.set('size', self._size_data_dimension)

                check_encoding_dtype(self.size_data)

                component = self._encodings.data[self._size_data_dimension].component
                categorical_data = get_categorical_data(self.size_data)

                if categorical_data is not None:
                    self._points[:, component] = categorical_data.cat.codes
                    self._size_categories = dict(zip(categorical_data, categorical_data.cat.codes))
                    self._size_histogram = get_histogram_from_df(categorical_data)
                else:
                    self._points[:, component] = self._size_norm(
                        zerofy_missing_values(self.size_data.values, 'Size')
                    )
                    self._size_histogram = get_histogram_from_df(
                        self._points[:, component],
                        self.get_histogram_bins("size"),
                        self.get_histogram_range("size")
                    )
                    self._size_categories = None

                if not self._encodings.data[self._size_data_dimension].prepared:
                    data_updated = True
                    # Make sure we don't prepare the data twice
                    self._encodings.data[self._size_data_dimension].prepared = True

        elif default is not UNDEF:
            # Presumably the user wants to switch to a static color encoding
            self._size_by = None
            self._size_histogram = None
            self._encodings.delete('size')

        if order is not UNDEF:
            if order is None or order == 'reverse':
                self._size_order = order
            elif self._size_categories is not None:
                # Define order of the sizes instead of changing `points[:, component_idx]`
                self._size_order = [self._size_categories[cat] for cat in self._size_order]

        if map is not UNDEF and map != 'auto' and map is not None:
            self._size_map_order = None
            if isinstance(map, tuple):
                # Assuming `map` is a triple specifying a linear space
                self._size_map = np.linspace(*map)
            elif isinstance(map, dict):
                # Assuming `map` is a dictionary of sizes
                self._size_map = list(map.values())
                self._size_map_order = list(map.keys())
                self._size_order = get_map_order(map, self._size_categories)
            else:
                self._size_map = np.asarray(map)

        if (self._size_map is None or map == 'auto') and self._size_by is not None:
            # The best we can do is provide a linear size map
            if self._size_categories is None:
                self._size_map = np.linspace(1, 10, 19)
            else:
                self._size_map = np.arange(1, len(self._size_categories) + 1)

        self._size_map = tolist(self._size_map)

        if self._size_categories is not None:
            assert len(self._size_categories) <= len(self._size_map), 'More categories than sizes'

        if labeling is not UNDEF:
            if labeling is None:
                self._size_labeling = None
            else:
                column = self._size_by if isinstance(self._size_by, str) else None
                self._size_labeling = create_labeling(labeling, column)

        # Update widget and encoding domain-range
        if self._size_by is not None and self._size_map is not None:
            final_size_map = order_map(self._size_map, self._size_order)
            self.update_widget('size', final_size_map)
            self._encodings.set_legend(
                'size',
                final_size_map,
                self._size_norm,
                self._size_categories,
                self._size_labeling,
                category_order=self._size_map_order,
            )
            self.update_widget('size_scale', get_scale(self, 'size'))
            self.update_widget('size_domain', get_domain(self, 'size'))
        else:
            self.update_widget('size', self._size)

        self.update_widget('size_histogram', self._size_histogram)
        self.update_widget('size_by', self.js_size_by)
        self.update_widget('size_title', self._size_by)
        self.update_widget('legend_encoding', self.get_legend_encoding())

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('prevent_filter_reset', True)
            self.update_widget('points', self.get_point_list())

        if any_not([default, by, map, norm, order, labeling], UNDEF):
            return self

        return dict(
            default = self._size,
            by = self._size_by,
            map = self._size_map,
            norm = self._size_norm,
            order = self._size_order,
            labeling = self._size_labeling,
        )

    @property
    def connect_by_data(self):
        if self._connect_by_data is not None:
            return self._connect_by_data
        return self._data[self._connect_by]

    @property
    def connect_order_data(self):
        if self._connect_order_data is not None:
            return self._connect_order_data
        return self._data[self._connect_order]

    def connect(
        self,
        by: Optional[Union[str, List[int], np.ndarray[int], None, Undefined]] = UNDEF,
        order: Optional[Union[List[int], np.ndarray[int], None, Undefined]] = UNDEF,
        **kwargs
    ):
        """
        Set or get the line-connection encoding of points.

        Parameters
        ----------
        by : str or array_like, optional
            The parameter determines which points should be connected by a line
            via shared indices. All points with the same line connection index
            will be connected. The indices can either be an array-like list of
            ints or a string referencing a column in the pd.DataFrame `data`.
        order : array_like, optional
            The ordering of the connected points. Without an order points are
            connected in the order they appear in data. The ordering overrides
            this behavior by providing a per-connection relative ordering. It
            must be specified as an array-like list of ints.
        kwargs : optional
            Options which can be used to skip updating the widget when
            `skip_widget_update` is set to `True`

        Returns
        -------
        self or dict
            If no parameter was provided the current point connection encoding
            is returned as a dictionary. Otherwise, `self` is returned.

        See Also
        --------
        connection_color : Set or get the connection color encoding.
        connection_opacity : Set or get the connection opacity encoding.
        connection_size : Set or get the connection size encoding.

        Examples
        --------
        >>> scatter.connect(by='group')
        <jscatter.jscatter.Scatter>

        >>> scatter.connect()
        {'by': 'group', 'order': None}
        """
        data_updated = False
        if by is not UNDEF:
            self._connect_by = by

            if by is not None:
                if isinstance(by, str):
                    self._connect_by_data = None
                else:
                    self._connect_by_data = pd.Series(
                        by,
                        index=self._data_index,
                        dtype='category'
                    )
                    self._connect_by = 'Custom Connect-By Data'

                categorical_data = get_categorical_data(self.connect_by_data)

                if categorical_data is not None:
                    self._points[:, COMPONENT_CONNECT] = categorical_data.cat.codes
                elif pd.api.types.is_integer_dtype(self.connect_by_data.dtype):
                    self._points[:, COMPONENT_CONNECT] = self.connect_by_data.values
                else:
                    raise ValueError('connect by only works with categorical data')

                data_updated = True

        if order is not UNDEF:
            self._connect_order = order

            if order is not None:
                if isinstance(order, str):
                    self._connect_order_data = None
                else:
                    self._connect_order_data = pd.Series(
                        order,
                        index=self._data_index,
                        dtype='int'
                    )
                    self._connect_order = 'Custom Connect-Order Data'

                if pd.api.types.is_integer_dtype(self.connect_order_data.dtype):
                    self._points[:, COMPONENT_CONNECT_ORDER] = self.connect_order_data
                else:
                    raise ValueError('connect order must be an integer type')

                data_updated = True

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('prevent_filter_reset', True)
            self.update_widget('points', self.get_point_list())

        self.update_widget('connect', bool(self._connect_by))

        if any_not([by, order], UNDEF):
            return self

        return dict(
            by = self._connect_by,
            order = self._connect_order,
        )

    @property
    def connection_color_data(self):
        if self._connection_color_data is not None:
            return self._connection_color_data
        return self._data[self._connection_color_by]

    def connection_color(
        self,
        default: Optional[Union[Color, Undefined]] = UNDEF,
        selected: Optional[Union[Color, Undefined]] = UNDEF,
        hover: Optional[Union[Color, Undefined]] = UNDEF,
        by: Optional[Union[Segment, str, List[float], np.ndarray, Undefined]] = UNDEF,
        map: Optional[Union[Auto, str, dict, list, LinearSegmentedColormap, ListedColormap, Undefined]] = UNDEF,
        norm: Optional[Union[Tuple[float, float], Normalize, Undefined]] = UNDEF,
        order: Optional[Union[Reverse, List[int], List[str], Undefined]] = UNDEF,
        labeling: Optional[Union[Labeling, Undefined]] = UNDEF,
        **kwargs
    ):
        """
        Set or get the color encoding of the point-connecting lines.

        Parameters
        ----------
        default : matplotlib compatible color, optional
            The color to be applied uniformly to all point-connecting lines.
        selected : matplotlib compatible color, optional
            The color to be applied uniformly to all point-connecting lines that
            contain at least one selected point.
        hover : matplotlib compatible color, optional
            The color to be applied uniformly to point-connecting lines that
            contain hovered points.
        by : str or array_like, optional
            The parameter is used for data-driven color encoding. It can either
            be an array-like list of values or a string referencing a column in
            the pd.DataFrame `data`.
            Additionally, this parameter can be set to `segment` to apply the
            coloring separately for each point-connecting line along the line
            segments. This can, for instance, be used to give the start of a
            line a different color than the end of a line.
        map : array_like, optional
            The color map used for data-driven color encoding. It can either be
            a string referencing a matplotlib color map, a matplotlib color map
            object, a list of matplotlib-compatible colors, a dictionary of
            category<->color pairs, or `auto`. When set to `auto`, jscatter will
            choose an appropriate color map.
        norm : array_like, optional
            The normalization method for data-driven color encoding. It can
            either a tuple defining a value range that maps to `[0, 1]` with
            `matplotlib.colors.Normalize` or a matplotlib normalizer instance.
        order : array_like, optional
            The order of the color map. It can either be a list of values (for
            categorical coloring) or `reverse` to reverse the color map.
        labeling : array_like, optional
            The labeling of the minimum and maximum value as well as the data
            variable. If defined, labels are shown with the legend. The labeling
            can either be a list of strings (`[minValue, maxValue, variable]`)
            or a dictionary (`{ min: label, max: label, variable: label }`).
        kwargs : optional
            Options which can be used to skip updating the widget when
            `skip_widget_update` is set to `True`

        Returns
        -------
        self or dict
            If no parameter was provided the current color encoding settings are
            returned as a dictionary. Otherwise, `self` is returned.

        Notes
        -----
        See https://matplotlib.org/3.5.0/tutorials/colors/colors.html for valid
        matplotlib colors.
        See https://matplotlib.org/3.5.0/api/colors_api.html for valid
        matplotlib normalizer classes.

        See Also
        --------
        connect : Set or get the line-connection encoding of points.
        connection_opacity : Set or get the connection opacity encoding.
        connection_size : Set or get the connection size encoding.

        Examples
        --------
        >>> scatter.connection_color('red')
        <jscatter.jscatter.Scatter>

        >>> scatter.connection_color(by='speed', map='plasma', order='reverse')
        <jscatter.jscatter.Scatter>

        >>> scatter.connection_color()
        {'color': (0, 0, 0, 0.66),
         'color_selected': (0, 0.55, 1, 1),
         'color_hover': (0, 0, 0, 1),
         'by': None,
         'map': None,
         'norm': <matplotlib.colors.Normalize at 0x12fa8b250>,
         'order': None,
         'labeling': None}
        """
        if default is not UNDEF:
            if default == 'inherit':
                self._connection_color = default
            else:
                try:
                    self._connection_color = to_rgba(default)
                except ValueError:
                    pass

        if selected is not UNDEF:
            try:
                self._connection_color_selected = to_rgba(selected)
                self.update_widget('connection_color_selected', self._connection_color_selected)
            except ValueError:
                pass

        if hover is not UNDEF:
            try:
                self._connection_color_hover = to_rgba(hover)
                self.update_widget('connection_color_hover', self._connection_color_hover)
            except ValueError:
                pass

        if norm is not UNDEF:
            if callable(norm):
                try:
                    self._connection_color_norm = norm
                except:
                    pass
            else:
                try:
                    vmin, vmax = norm
                    self._connection_color_norm = Normalize(vmin, vmax)
                except:
                    if norm is None:
                        self._connection_color_norm = create_default_norm()
                    pass

        data_updated = True
        if by is not UNDEF:
            self._connection_color_by = by

            if by is None:
                self._encodings.delete('connection_color')

            elif by == 'segment':
                pass

            elif by == 'inherit':
                pass

            else:
                if isinstance(by, str):
                    self._connection_color_data = None
                else:
                    self._connection_color_data = pd.Series(
                        by,
                        index=self._data_index,
                    )
                    self._connection_color_by = 'Custom Connection-Color Data'

                self._encodings.set('connection_color', self._connection_color_data_dimension)

                check_encoding_dtype(self.connection_color_data)

                component = self._encodings.data[self._connection_color_data_dimension].component
                categorical_data = get_categorical_data(self.connection_color_data)

                if categorical_data is not None:
                    self._connection_color_categories = dict(zip(categorical_data, categorical_data.cat.codes))
                    self._points[:, component] = categorical_data.cat.codes
                else:
                    self._connection_color_categories = None
                    self._points[:, component] = self._connection_color_norm(
                        zerofy_missing_values(self.connection_color_data.values, 'Connection color')
                    )

                if not self._encodings.data[self._connection_color_data_dimension].prepared:
                    data_updated = True
                    # Make sure we don't prepare the data twice
                    self._encodings.data[self._connection_color_data_dimension].prepared = True

        elif default is not UNDEF:
            # Presumably the user wants to switch to a static color encoding
            self._connection_color_by = None
            self._encodings.delete('connection_color')

        if order is not UNDEF:
            if order is None or order == 'reverse':
                self._connection_color_order = order
            elif self._connection_color_categories is not None:
                # Define order of the colors instead of changing `points[:, component_idx]`
                self._connection_color_order = [self._connection_color_categories[cat] for cat in order]

        if map is not UNDEF and map != 'auto' and map is not None:
            if self._connection_color_categories is None:
                if callable(map):
                    # Assuming `map` is a Matplotlib LinearSegmentedColormap
                    self._connection_color_map = map(range(256)).tolist()
                elif isinstance(map, str):
                    # Assuming `map` is the name of a Matplotlib LinearSegmentedColormap
                    self._connection_color_map = plt.get_cmap(map)(range(256)).tolist()
                else:
                    # Assuming `map` is a list of colors
                    self._connection_color_map = [to_rgba(c) for c in map]
            else:
                self._connection_color_map_order = None
                if callable(map):
                    # Assuming `map` is a Matplotlib ListedColormap
                    self._connection_color_map = [to_rgba(c) for c in map.colors]
                elif isinstance(map, str):
                    # Assuming `map` is the name of a Matplotlib ListedColormap
                    self._connection_color_map = [to_rgba(c) for c in plt.get_cmap(map).colors]
                elif isinstance(map, dict):
                    # Assuming `map` is a dictionary of colors
                    self._connection_color_map = [to_rgba(c) for c in map.values()]
                    self._connection_color_map_order = list(map.keys())
                    self._connection_color_order = get_map_order(map, self._connection_color_categories)
                else:
                    # Assuming `map` is a list of colors
                    self._connection_color_map = [to_rgba(c) for c in map]

        if (self._connection_color_map is None or map == 'auto') and self._connection_color_by is not None:
            # Assign default color maps
            if self._connection_color_categories is None:
                self._connection_color_map = plt.get_cmap('viridis')(range(256)).tolist()
            elif len(self._connection_color_categories) > 8:
                if self._background_color_luminance < 0.5:
                    self._connection_color_map = glasbey_light
                else:
                    self._connection_color_map = glasbey_dark
            else:
                self._connection_color_map = okabe_ito

        if self._connection_color_categories is not None:
            assert len(self._connection_color_categories) <= len(self._connection_color_map), 'More categories than connection colors'

        if labeling is not UNDEF:
            if labeling is None:
                self._connection_color_labeling = None
            else:
                column = self._connection_color_by if isinstance(self._connection_color_by, str) else None
                self._connection_color_labeling = create_labeling(labeling, column)

        # Update widget and legend encoding
        if self._connection_color_by is not None and self._connection_color_map is not None:
            final_connection_color_map = order_map(
                self._connection_color_map,
                self._connection_color_order,
            )
            self.update_widget('connection_color', final_connection_color_map)
            self._encodings.set_legend(
                'connection_color',
                final_connection_color_map,
                self._connection_color_norm,
                self._connection_color_categories,
                self._connection_color_labeling,
                category_order=self._connection_color_map_order,
            )
        else:
            self.update_widget('connection_color', self._connection_color)

        self.update_widget('connection_color_by', self.js_connection_color_by)
        self.update_widget('legend_encoding', self.get_legend_encoding())

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('prevent_filter_reset', True)
            self.update_widget('points', self.get_point_list())

        if any_not([default, selected, hover, by, map, norm, order, labeling], UNDEF):
            return self

        return dict(
            default = self._connection_color,
            selected = self._connection_color_selected,
            hover = self._connection_color_hover,
            by = self._connection_color_by,
            map = self._connection_color_map,
            norm = self._connection_color_norm,
            order = self._connection_color_order,
            labeling = self._connection_color_labeling,
        )

    @property
    def connection_opacity_data(self):
        if self._connection_opacity_data is not None:
            return self._connection_opacity_data
        return self._data[self._connection_opacity_by]

    def connection_opacity(
        self,
        default: Optional[Union[float, Undefined]] = UNDEF,
        by: Optional[Union[str, List[float], np.ndarray, Undefined]] = UNDEF,
        map: Optional[Union[Auto, dict, List[float], Tuple[float, float, int], Undefined]] = UNDEF,
        norm: Optional[Union[Tuple[float, float], Normalize, Undefined]] = UNDEF,
        order: Optional[Union[Reverse, List[int], List[str], Undefined]] = UNDEF,
        labeling: Optional[Union[Labeling, Undefined]] = UNDEF,
        **kwargs
    ):
        """
        Set or get the opacity encoding of the point-connecting lines.

        Parameters
        ----------
        default : float, optional
            The opacity to be applied uniformly to all point-connecting lines.
        by : str or array_like, optional
            The parameter is used for data-driven opacity encoding. It can
            either be an array-like list of floats or a string referencing a
            column in the pd.DataFrame `data`.
        map : array_like, optional
            The opacity map used for data-driven opacity encoding. It can either
            be a list of floats, a dictionary of category<->opacity pairs, a
            triple specifying a `np.linspace`, or `auto`. When set to `auto`,
            jscatter will choose an appropriate opacity map.
        norm : array_like, optional
            The normalization method for data-driven opacity encoding. It can
            either a tuple defining a value range that maps to `[0, 1]` with
            `matplotlib.colors.Normalize` or a matplotlib normalizer instance.
        order : array_like, optional
            The order of the opacity map. It can either be a list of values (for
            categorical opacity values) or `reverse` to reverse the opacity map.
        labeling : array_like, optional
            The labeling of the minimum and maximum value as well as the data
            variable. If defined, labels are shown with the legend. The labeling
            can either be a list of strings (`[minValue, maxValue, variable]`)
            or a dictionary (`{ min: label, max: label, variable: label }`).
        kwargs : optional
            Options which can be used to skip updating the widget when
            `skip_widget_update` is set to `True`

        Returns
        -------
        self or dict
            If no parameter was provided the current opacity encoding settings
            are returned as a dictionary. Otherwise, `self` is returned.

        Notes
        -----
        See https://matplotlib.org/3.5.0/api/colors_api.html for valid
        matplotlib normalizer classes.
        See https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        for how the triple (start, stop, num) specifies a linear space

        See Also
        --------
        connect : Set or get the line-connection encoding of points.
        connection_color : Set or get the connection color encoding.
        connection_size : Set or get the connection size encoding.

        Examples
        --------
        >>> scatter.connection_opacity(0.1)
        <jscatter.jscatter.Scatter>

        >>> scatter.connection_opacity(by='speed', map=(0, 1, 5))
        <jscatter.jscatter.Scatter>

        >>> scatter.connection_opacity()
        {'opacity': 0.1,
         'by': None,
         'map': None,
         'norm': <matplotlib.colors.Normalize at 0x12fa8b700>,
         'order': None,
         'labeling': None}
        """
        if default is not UNDEF:
            try:
                self._connection_opacity = float(default)
                assert self._connection_opacity >= 0 and self._connection_opacity <= 1, 'Connection opacity must be in [0,1]'
                self.update_widget('connection_opacity', self._connection_opacity_map or self._connection_opacity)
            except ValueError:
                pass

        if norm is not UNDEF:
            if callable(norm):
                try:
                    self._connection_opacity_norm.clip = norm
                except:
                    pass
            else:
                try:
                    vmin, vmax = norm
                    self._connection_opacity_norm = Normalize(vmin, vmax)
                except:
                    if norm is None:
                        self._connection_opacity_norm = create_default_norm()
                    pass

        data_updated = False
        if by is not UNDEF:
            self._connection_opacity_by = by

            if by is None:
                self._encodings.delete('connection_opacity')

            else:
                if isinstance(by, str):
                    self._connection_opacity_data = None
                else:
                    self._connection_opacity_data = pd.Series(
                        by,
                        index=self._data_index,
                    )
                    self._connection_opacity_by = 'Custom Connection-Opacity Data'

                self._encodings.set('connection_opacity', self._connection_opacity_data_dimension)

                check_encoding_dtype(self.connection_opacity_data)

                component = self._encodings.data[self._connection_opacity_data_dimension].component
                categorical_data = get_categorical_data(self.connection_opacity_data)

                if categorical_data is not None:
                    self._points[:, component] = categorical_data.cat.codes
                    self._connection_opacity_categories = dict(zip(categorical_data, categorical_data.cat.codes))
                else:
                    self._points[:, component] = self._connection_opacity_norm(
                        zerofy_missing_values(self.connection_opacity_data.values, 'Connection opacity')
                    )
                    self._connection_opacity_categories = None

                if not self._encodings.data[self._connection_opacity_data_dimension].prepared:
                    data_updated = True
                    # Make sure we don't prepare the data twice
                    self._encodings.data[self._connection_opacity_data_dimension].prepared = True

        elif default is not UNDEF:
            # Presumably the user wants to switch to a static opacity encoding
            self._connection_opacity_by = None
            self._encodings.delete('connection_opacity')

        if order is not UNDEF:
            if order is None or order == 'reverse':
                self._connection_opacity_order = order
            elif self._connection_opacity_categories is not None:
                # Define order of the opacities instead of changing `points[:, component_idx]`
                self._connection_opacity_order = [
                    self._connection_opacity_categories[cat] for cat in order
                ]

        if map is not UNDEF and map != 'auto' and map is not None:
            self._connection_opacity_map_order = None
            if type(map) == tuple:
                # Assuming `map` is a triple specifying a linear space
                self._connection_opacity_map = np.linspace(*map)
            elif isinstance(map, dict):
                # Assuming `map` is a dictionary of opacities
                self._connection_opacity_map = list(map.values())
                self._connection_opacity_map_order = list(map.keys())
                self._connection_opacity_order = get_map_order(map, self._connection_opacity_categories)
            else:
                self._connection_opacity_map = np.asarray(map)

        if (self._connection_opacity_map is None or map == 'auto') and self._connection_opacity_by is not None:
            # The best we can do is provide a linear opacity map
            if self._connection_opacity_categories is not None:
                self._connection_opacity_map = np.linspace(
                    1 / len(self._connection_opacity_categories),
                    1,
                    len(self._connection_opacity_categories)
                )
            else:
                self._connection_opacity_map = np.linspace(1/256, 1, 256)

        self._connection_opacity_map = tolist(self._connection_opacity_map)

        if self._connection_opacity_categories is not None:
            assert len(self._connection_opacity_categories) <= len(self._connection_opacity_map), 'More categories than connection opacities'

        if labeling is not UNDEF:
            if labeling is None:
                self._connection_opacity_labeling = None
            else:
                column = self._connection_opacity_by if isinstance(self._connection_opacity_by, str) else None
                self._connection_opacity_labeling = create_labeling(labeling, column)

        # Update widget and legend encoding
        if self._connection_opacity_by is not None and self._connection_opacity_map is not None:
            final_opacity_map = order_map(
                self._connection_opacity_map,
                self._connection_opacity_order,
            )
            self.update_widget('connection_opacity', final_opacity_map)
            self._encodings.set_legend(
                'connection_opacity',
                final_opacity_map,
                self._connection_opacity_norm,
                self._connection_opacity_categories,
                self._connection_opacity_labeling,
                category_order=self._connection_opacity_map_order,
            )
        else:
            self.update_widget('connection_opacity', self._connection_opacity)

        self.update_widget('connection_opacity_by', self.js_connection_opacity_by)
        self.update_widget('legend_encoding', self.get_legend_encoding())

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('prevent_filter_reset', True)
            self.update_widget('points', self.get_point_list())

        if any_not([default, by, map, norm, order, labeling], UNDEF):
            return self

        return dict(
            default = self._connection_opacity,
            by = self._connection_opacity_by,
            map = self._connection_opacity_map,
            norm = self._connection_opacity_norm,
            order = self._connection_opacity_order,
            labeling = self._connection_opacity_labeling,
        )

    @property
    def connection_size_data(self):
        if self._connection_size_data is not None:
            return self._connection_size_data
        return self._data[self._connection_size_by]

    def connection_size(
        self,
        default: Optional[Union[float, Undefined]] = UNDEF,
        by: Optional[Union[str, List[float], np.ndarray, Undefined]] = UNDEF,
        map: Optional[Union[Auto, dict, List[float], Tuple[float, float, int], Undefined]] = UNDEF,
        norm: Optional[Union[Tuple[float, float], Normalize, Undefined]] = UNDEF,
        order: Optional[Union[Reverse, List[int], List[str], Undefined]] = UNDEF,
        labeling: Optional[Union[Labeling, Undefined]] = UNDEF,
        **kwargs
    ):
        """
        Set or get the size encoding of the point-connecting lines.

        Parameters
        ----------
        default : float, optional
            The size to be applied uniformly to all point-connecting lines.
        by : str or array_like, optional
            The parameter is used for data-driven size encoding. It can
            either be an array-like list of floats or a string referencing a
            column in the pd.DataFrame `data`.
        map : array_like, optional
            The size map used for data-driven size encoding. It can either
            be a list of floats, a dictionary of category<->size pairs, a
            triple specifying a `np.linspace`, or `auto`. When set to `auto`,
            jscatter will choose an appropriate size map.
        norm : array_like, optional
            The normalization method for data-driven size encoding. It can
            either a tuple defining a value range that maps to `[0, 1]` with
            `matplotlib.colors.Normalize` or a matplotlib normalizer instance.
        order : array_like, optional
            The order of the size map. It can either be a list of values (for
            categorical size values) or `reverse` to reverse the size map.
        labeling : array_like, optional
            The labeling of the minimum and maximum value as well as the data
            variable. If defined, labels are shown with the legend. The labeling
            can either be a list of strings (`[minValue, maxValue, variable]`)
            or a dictionary (`{ min: label, max: label, variable: label }`).
        kwargs : optional
            Options which can be used to skip updating the widget when
            `skip_widget_update` is set to `True`

        Returns
        -------
        self or dict
            If no parameter was provided the current size encoding settings
            are returned as a dictionary. Otherwise, `self` is returned.

        Notes
        -----
        See https://matplotlib.org/3.5.0/api/colors_api.html for valid
        matplotlib normalizer classes.
        See https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        for how the triple (start, stop, num) specifies a linear space

        See Also
        --------
        connect : Set or get the line-connection encoding of points.
        connection_color : Set or get the connection color encoding.
        connection_opacity : Set or get the connection opacity encoding.

        Examples
        --------
        >>> scatter.connection_size(2)
        <jscatter.jscatter.Scatter>

        >>> scatter.connection_size(by='group_size', map=(2, 20, 4))
        <jscatter.jscatter.Scatter>

        >>> scatter.connection_size()
        {'size': 2,
         'by': None,
         'map': None,
         'norm': <matplotlib.colors.Normalize at 0x12fa8b7c0>,
         'order': None,
         'labeling': None}
        """
        if default is not UNDEF:
            try:
                self._connection_size = int(default)
                assert self._connection_size > 0, 'Connection size must be a positive integer'
            except ValueError:
                pass

        if norm is not UNDEF:
            if callable(norm):
                try:
                    self._connection_size_norm = norm
                except:
                    pass
            else:
                try:
                    vmin, vmax = norm
                    self._connection_size_norm = Normalize(vmin, vmax)
                except:
                    if norm is None:
                        self._connection_size_norm = create_default_norm()
                    pass

        data_updated = False
        if by is not UNDEF:
            self._connection_size_by = by

            if by is None:
                self._encodings.delete('connection_size')

            else:
                if isinstance(by, str):
                    self._connection_size_data = None
                else:
                    self._connection_size_data = pd.Series(
                        by,
                        index=self._data_index,
                    )
                    self._connection_size_by = 'Custom Connection-Size Data'

                self._encodings.set('connection_size', self._connection_size_data_dimension)

                check_encoding_dtype(self.connection_size_data)

                component = self._encodings.data[self._connection_size_data_dimension].component
                categorical_data = get_categorical_data(self.connection_size_data)

                if categorical_data is not None:
                    self._points[:, component] = categorical_data.cat.codes
                    self._connection_size_categories = dict(zip(categorical_data, categorical_data.cat.codes))
                else:
                    self._points[:, component] = self._connection_size_norm(
                        zerofy_missing_values(self.connection_size_data.values, 'Connection size')
                    )
                    self._connection_size_categories = None

                if not self._encodings.data[self._connection_size_data_dimension].prepared:
                    data_updated = True
                    # Make sure we don't prepare the data twice
                    self._encodings.data[self._connection_size_data_dimension].prepared = True

        elif default is not UNDEF:
            # Presumably the user wants to switch to a static size encoding
            self._connection_size_by = None
            self._encodings.delete('connection_size')

        if order is not UNDEF:
            if order is None or order == 'reverse':
                self._connection_size_order = order
            elif self._connection_size_categories is not None:
                # Define order of the sizes instead of changing `points[:, component_idx]`
                self._connection_size_order = [self._connection_size_categories[cat] for cat in order]

        if map is not UNDEF and map != 'auto' and map is not None:
            self._connection_size_map_order = None
            if type(map) == tuple:
                # Assuming `map` is a triple specifying a linear space
                self._connection_size_map = np.linspace(*map)
            elif isinstance(map, dict):
                # Assuming `map` is a dictionary of sizes
                self._connection_size_map = list(map.values())
                self._connection_size_map_order = list(map.keys())
                self._connection_size_order = get_map_order(map, self._connection_size_categories)
            else:
                self._connection_size_map = np.asarray(map)

        if (self._connection_size_map is None or map == 'auto') and self._connection_size_by is not None:
            # The best we can do is provide a linear size map
            if self._connection_size_categories is None:
                self._connection_size_map = np.linspace(1, 10, 19)
            else:
                self._connection_size_map = np.arange(1, len(self._connection_size_categories) + 1)

        self._connection_size_map = tolist(self._connection_size_map)

        if labeling is not UNDEF:
            if labeling is None:
                self._connection_size_labeling = None
            else:
                column = self._connection_size_by if isinstance(self._connection_size_by, str) else None
                self._connection_size_labeling = create_labeling(labeling, column)

        # Update widget and legend encoding
        if self._connection_size_by is not None and self._connection_size_map is not None:
            final_connection_size_map = order_map(
                self._connection_size_map,
                self._connection_size_order
            )
            self.update_widget('connection_size', final_connection_size_map)
            self._encodings.set_legend(
                'connection_size',
                final_connection_size_map,
                self._connection_size_norm,
                self._connection_size_categories,
                self._connection_size_labeling,
                category_order=self._connection_size_map_order,
            )
        else:
            self.update_widget('connection_size', self._connection_size)

        self.update_widget('connection_size_by', self.js_connection_size_by)
        self.update_widget('legend_encoding', self.get_legend_encoding())

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('prevent_filter_reset', True)
            self.update_widget('points', self.get_point_list())

        if self._connection_size_categories is not None:
            assert len(self._connection_size_categories) <= len(self._connection_size_map), 'More categories than connection sizes'

        if any_not([default, by, map, norm, order, labeling], UNDEF):
            return self

        return dict(
            default = self._connection_size,
            by = self._connection_size_by,
            map = self._connection_size_map,
            norm = self._connection_size_norm,
            order = self._connection_size_order,
            labeling = self._connection_size_labeling,
        )

    def background(
        self,
        color: Optional[Union[Color, Undefined]] = UNDEF,
        image: Optional[Union[str, bytes, Undefined]] = UNDEF,
        **kwargs
    ):
        """
        Set or get the scatter plot's background.

        Parameters
        ----------
        color : matplotlib compatible color, optional
            The background color of the scatter plot. The value must be a
            Matplotlib-compatible color.
        image : array_like, optional
            The background image of the scatter plot. It can either be a URL
            string pointing to an image or a PIL image object that is
            understood by `matplotlib.pyplot.imshow`
        kwargs : optional
            Options to be passed to `matplotlib.pyplot.imshow`

        Returns
        -------
        self or dict
            If no parameter was provided the current background settings are
            returned as a dictionary. Otherwise, `self` is returned.

        Notes
        -----
        See https://matplotlib.org/3.5.0/tutorials/colors/colors.html for valid
        matplotlib colors.
        See https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.imshow.html
        for acceptable images and options.

        See Also
        --------
        width : Set or get the width of the scatter plot.
        height : Set or get the height of the scatter plot.

        Examples
        --------
        >>> scatter.background(color='red')
        <jscatter.jscatter.Scatter>

        >>> scatter.background(image='https://picsum.photos/200/200/?random')
        <jscatter.jscatter.Scatter>

        >>> scatter.background()
        {'color': (1.0, 1.0, 1.0, 1.0), 'image': None}
        """
        if color is not UNDEF:
            try:
                self._background_color = to_rgba(color)
                self.update_widget('background_color', self._background_color)
            except:
                if color is None:
                    self._background_color = to_rgba(default_background_color)
                    self.update_widget('background_color', self._background_color)
                pass

            self._background_color_luminance = math.sqrt(
                0.299 * self._background_color[0] ** 2
                + 0.587 * self._background_color[1] ** 2
                + 0.114 * self._background_color[2] ** 2
            )

            self.update_widget('reticle_color', self.get_reticle_color())
            self.update_widget('axes_color', self.get_axes_color())
            self.update_widget('legend_color', self.get_legend_color())
            self.update_widget('tooltip_color', self.get_tooltip_color())

        if image is not UNDEF:
            if uri_validator(image):
                self._background_image = image
                self.update_widget('background_image', self._background_image)

            else:
                try:
                    im = plt.imshow(image, **kwargs)

                    x = im.make_image()
                    h, w, d = x.as_rgba_str()
                    self._background_image = np.fromstring(d, dtype=np.uint8).reshape(h, w, 4)
                    self.update_widget('background_image', self._background_image)
                except:
                    if image is None:
                        self._background_image = None
                        self.update_widget('background_image', self._background_image)
                    pass

        if any_not([color, image], UNDEF):
            return self

        return dict(
            color = self._background_color,
            image = self._background_image,
        )

    def camera(
        self,
        target: Optional[Union[Tuple[float, float], Undefined]] = UNDEF,
        distance: Optional[Union[float, Undefined]] = UNDEF,
        rotation: Optional[Union[float, Undefined]] = UNDEF,
        view: Optional[Union[List[float], np.ndarray, Undefined]] = UNDEF,
    ):
        """
        Set or get the camera settings.

        Parameters
        ----------
        target : tuple of float, optional
            The camera target point on the 2D scatter plot plane.
        distance : float, optional
            The distance of the camera to the 2D scatter plot plane. The smaller
            the value the more zoomed in the view is.
        rotation : float, optional
            The camera rotation in radians.
        view : float, optional
            Columnar 4x4 camera's view matrix. The matrix is an array-like list
            of 16 floats where the first numbers represent the first column etc.

        Returns
        -------
        self or dict
            If no parameter was provided the current camera settings are
            returned as a dictionary. Otherwise, `self` is returned.

        Notes
        -----
        All parameters must be in normalized device coordinates! See
        https://github.com/flekschas/dom-2d-camera for context.

        Examples
        --------
        >>> scatter.camera(target=[0.5, 0.5])
        <jscatter.jscatter.Scatter>

        >>> scatter.camera(distance=2)
        <jscatter.jscatter.Scatter>

        >>> scatter.camera(rotation=0.5)
        <jscatter.jscatter.Scatter>

        >>> scatter.camera(view=[
        >>>     2, 0, 0, 0,
        >>>     0, 2, 0, 0,
        >>>     0, 0, 1, 0,
        >>>     1, 1, 0, 1
        >>> ])
        <jscatter.jscatter.Scatter>

        >>> scatter.camera()
        {'target': [0, 0], 'distance': 1.0, 'rotation': 0.0, 'view': None}
        """
        if target is not UNDEF:
            self._camera_target = target
            self.update_widget('camera_target', self._camera_target)

        if distance is not UNDEF:
            try:
                self._camera_distance = float(distance)
                assert self._camera_distance > 0, 'Camera distance must be positive'
                self.update_widget('camera_distance', self._camera_distance)
            except ValueError:
                pass

        if rotation is not UNDEF:
            try:
                self._camera_rotation = float(rotation)
                self.update_widget('camera_rotation', self._camera_rotation)
            except ValueError:
                pass

        if view is not UNDEF:
            self._camera_view = view
            self.update_widget('camera_view', self._camera_view)

        if any_not([target, distance, rotation, view], UNDEF):
            return self

        return dict(
            target = self._camera_target,
            distance = self._camera_distance,
            rotation = self._camera_rotation,
            view = self._camera_view,
        )

    def lasso(
        self,
        color: Optional[Union[Color, Undefined]] = UNDEF,
        initiator: Optional[Union[bool, Undefined]] = UNDEF,
        min_delay: Optional[Union[int, Undefined]] = UNDEF,
        min_dist: Optional[Union[float, Undefined]]  = UNDEF,
        on_long_press: Optional[Union[bool, Undefined]]  = UNDEF,
    ):
        """
        Set or get the lasso settings.

        Parameters
        ----------
        color : matplotlib compatible color, optional
            The lasso color
        initiator : bool, optional
            When set to `True`, the lasso can be initiated via a click on the
            background and then click+hold+drag onto the circle with the dashed
            outline that appears. This circle, the "lasso initiator", can also
            be triggered via a long press anywhere.
        min_delay : float, optional
            The minimum delay in milliseconds before the lasso polygon is
            extended. In 99.99% of the cases, you can ignore this setting but
            in can be useful to lower the delay if you want a precise lasso when
            you move your mouse quickly.
        min_dist : float, optional
            The minimum distance from the previous mouse position before the
            lasso polygon is extended. In 99.99% of the cases, you can ignore
            this setting but in can be useful to lower the distance if you want
            a high-resolution lasso.
        on_long_press : bool, optional
            When set to `True`, the lasso is activated upon a long press.

        Returns
        -------
        self or dict
            If no parameter was provided the current lasso settings are
            returned as a dictionary. Otherwise, `self` is returned.

        Notes
        -----
        See https://user-images.githubusercontent.com/932103/106489598-f42c4480-6482-11eb-8286-92a9956e1d20.gif
        for an example of how the lasso initiator works

        See Also
        --------
        mouse : Set or get the mouse mode.

        Examples
        --------
        >>> scatter.lasso(color='red')
        <jscatter.jscatter.Scatter>

        >>> scatter.lasso(initiator=False)
        <jscatter.jscatter.Scatter>

        >>> scatter.lasso(min_delay=5)
        <jscatter.jscatter.Scatter>

        >>> scatter.lasso(min_dist=2)
        <jscatter.jscatter.Scatter>

        >>> scatter.lasso(on_long_press=False)
        <jscatter.jscatter.Scatter>

        >>> scatter.lasso()
        {'color': (0, 0.666666667, 1, 1),
         'initiator': False,
         'min_delay': 10,
         'min_dist': 3,
         'on_long_press': True}
        """
        if color is not UNDEF:
            try:
                self._lasso_color = to_rgba(color)
                self.update_widget('lasso_color', self._lasso_color)
            except:
                pass

        if initiator is not UNDEF:
            try:
                self._lasso_initiator = bool(initiator)
                self.update_widget('lasso_initiator', self._lasso_initiator)
            except:
                pass

        if min_delay is not UNDEF:
            try:
                self._lasso_min_delay = to_rgba(color)
                self.update_widget('lasso_min_delay', self._lasso_min_delay)
            except:
                pass

        if min_dist is not UNDEF:
            try:
                self._lasso_min_dist = float(min_dist)
                self.update_widget('lasso_min_dist', self._lasso_min_dist)
            except:
                pass

        if on_long_press is not UNDEF:
            try:
                self._lasso_on_long_press = bool(on_long_press)
                self.update_widget('lasso_on_long_press', self._lasso_on_long_press)
            except:
                pass

        if any_not([color, initiator, min_delay, min_dist], UNDEF):
            return self

        return dict(
            color = self._lasso_color,
            initiator = self._lasso_initiator,
            min_delay = self._lasso_min_delay,
            min_dist = self._lasso_min_dist,
            on_long_press = self._lasso_on_long_press,
        )

    def width(
        self,
        width: Optional[Union[Auto, int, Undefined]] = UNDEF
    ):
        """
        Set or get the width of the scatter plot.

        Parameters
        ----------
        width : int or 'auto', optional
            The width of the scatter plot in pixel. When set to `'auto'` the
            width is bound to the width of the notebook cell the scatter plot
            is rendered in.

        Returns
        -------
        self or dict
            If no width was provided the current width is returned. Otherwise,
            `self` is returned.

        See Also
        --------
        height : Set or get the height of the scatter plot

        Examples
        --------
        >>> scatter.width(512)
        <jscatter.jscatter.Scatter>

        >>> scatter.width()
        'auto'
        """
        if width is not UNDEF:
            try:
                self._width = int(width)
                self.update_widget('width', self._width)
            except:
                if width == 'auto':
                    self._width = width
                    self.update_widget('width', self._width)

                pass

            return self

        return self._width

    def height(self, height: Optional[Union[int, Undefined]] = UNDEF):
        """
        Set or get the height of the scatter plot.

        Parameters
        ----------
        width : int, optional
            The height of the scatter plot in pixel.

        Returns
        -------
        self or dict
            If no height was provided the current height is returned. Otherwise,
            `self` is returned.

        See Also
        --------
        width : Set or get the width of the scatter plot

        Examples
        --------
        >>> scatter.height(512)
        <jscatter.jscatter.Scatter>

        >>> scatter.height()
        240
        """
        if height is not UNDEF:
            try:
                self._height = int(height)
                self.update_widget('height', self._height)
            except:
                pass

            return self

        return self._height

    def get_reticle_color(self):
        try:
            return to_rgba(self._reticle_color)
        except ValueError:
            if self._background_color_luminance < 0.25:
                return (1, 1, 1, 0.15)
            elif self._background_color_luminance < 0.5:
                return (1, 1, 1, 0.23)
            elif self._background_color_luminance < 0.75:
                return (0, 0, 0, 0.2)

            return (0, 0, 0, 0.1) # Defaut

    def get_axes_color(self):
        if self._background_color_luminance < 0.5:
            return (1, 1, 1, 1)

        return (0, 0, 0, 1)

    def get_legend_color(self):
        if self._background_color_luminance < 0.5:
            return (1, 1, 1, 1)

        return (0, 0, 0, 1)

    def get_tooltip_color(self):
        if self._background_color_luminance < 0.5:
            return (0.2, 0.2, 0.2, 1)

        return (1, 1, 1, 1)

    def get_legend_encoding(self):
        return {
            channel: self._encodings.get_legend(channel)
            for channel
            in self._encodings.visual.keys()
        }

    def reticle(
        self,
        show: Optional[Union[bool, Undefined]] = UNDEF,
        color: Optional[Union[Color, Undefined]] = UNDEF
    ):
        """
        Set or get the reticle setting.

        Parameters
        ----------
        show : bool, optional
            When set to `True` a reticle will be shown over hovered points.
        color : matplotlib compatible color, optional
            The reticle color.

        Returns
        -------
        self or dict
            If no parameters are provided the current reticle settings are
            returned as a dictionary. Otherwise, `self` is returned.

        Examples
        --------
        >>> scatter.reticle(show=False)
        <jscatter.jscatter.Scatter>

        >>> scatter.reticle(color='red')
        <jscatter.jscatter.Scatter>

        >>> scatter.reticle()
        {'show': True, 'color': 'auto'}
        """
        if show is not UNDEF:
            try:
                self._reticle = bool(show)
                self.update_widget('reticle', self._reticle)
            except:
                pass

        if color is not UNDEF:
            if color == 'auto':
                self._reticle_color = 'auto'
                self.update_widget('reticle_color', self.get_reticle_color())
            else:
                try:
                    self._reticle_color = to_rgba(color)
                    self.update_widget('reticle_color', self.get_reticle_color())
                except:
                    pass

        if any_not([show, color], UNDEF):
            return self

        return dict(
            show = self._reticle,
            color = self._reticle_color,
        )

    def mouse(
        self,
        mode: Optional[Union[MouseModes, Undefined]] = UNDEF
    ):
        """
        Set or get the mouse mode.

        Parameters
        ----------
        show : 'panZoom', 'lasso' or 'rotate', optional
            The mouse mode. Currently, three modes are supported: pan & zoom,
            lasso selection, or rotating the scatter plot.

        Returns
        -------
        self or dict
            If no mode is provided the current mouse mode is returned.
            Otherwise, `self` is returned.

        Examples
        --------
        >>> scatter.mouse('lasso')
        <jscatter.jscatter.Scatter>

        >>> scatter.mouse()
        'panZoom'
        """
        if mode is not UNDEF:
            try:
                self._mouse_mode = mode
                self.update_widget('mouse_mode', mode)
            except:
                pass

            return self

        return self._mouse_mode

    def axes(
        self,
        axes: Optional[Union[bool, Undefined]] = UNDEF,
        grid: Optional[Union[bool, Undefined]] = UNDEF,
        labels: Optional[Union[bool, list, dict, Undefined]] = UNDEF,
    ):
        """
        Set or get the axes settings.

        Parameters
        ----------
        axes : bool, optional
            When set to `True`, an x and y axis will be shown.
        grid : bool, optional
            When set to `True`, the x and y tick marks are extended into a grid.
        labels : bool or list of str or dict of str, optional
            When set to `True`, the x and y axes labels are the x and y column
            names. Alternatively pass a list of strings or a dictionary with two
            keys (x and y).

        Returns
        -------
        self or dict
            If no parameters are provided the current axes settings are
            returned as a dictionary. Otherwise, `self` is returned.

        Examples
        --------
        >>> scatter.axes(False)
        <jscatter.jscatter.Scatter>

        >>> scatter.axes(grid=True)
        <jscatter.jscatter.Scatter>

        >>> scatter.axes(labels=['x', 'y'])
        <jscatter.jscatter.Scatter>

        >>> scatter.axes()
        {'axes': True, 'grid': True, 'labels': ['x', 'y']}
        """
        if axes is not UNDEF:
            try:
                self._axes = axes
                self.update_widget('axes', axes)
            except:
                pass

        if grid is not UNDEF:
            try:
                self._axes_grid = grid
                self.update_widget('axes_grid', grid)
            except:
                pass

        if labels is not UNDEF:
            if isinstance(labels, bool):
                self._axes_labels = labels
            elif isinstance(labels, dict):
                self._axes_labels = [
                    labels.get('x', ''), labels.get('y', '')
                ]
            else:
                self._axes_labels = labels

            try:
                self.update_widget('axes_labels', self.get_axes_labels())
            except:
                pass

        if any_not([axes, grid, labels], UNDEF):
            return self

        return dict(
            axes = self._axes,
            grid = self._axes_grid,
            labels = self._axes_labels,
        )

    def legend(
        self,
        legend: Optional[Union[bool, Undefined]] = UNDEF,
        position: Optional[Union[LegendPosition, Undefined]] = UNDEF,
        size: Optional[Union[Size, Undefined]] = UNDEF,
    ):
        """
        Set or get the legend settings.

        Parameters
        ----------
        legend : bool, optional
            When set to `True`, a legend will be shown.
        position : str, optional
            The legend position. Must be one of top, left, right, bottom,
            top-left, top-right, bottom-left, bottom-right, or center.
        size : small or medium or large, optional
            The size of the legend. Must be one of small, medium, or large

        Returns
        -------
        self or dict
            If no parameters are provided the current legend settings are
            returned as a dictionary. Otherwise, `self` is returned.

        Examples
        --------
        >>> scatter.legend(True)
        <jscatter.jscatter.Scatter>

        >>> scatter.legend(position='top-right')
        <jscatter.jscatter.Scatter>

        >>> scatter.legend(size='large')
        <jscatter.jscatter.Scatter>

        >>> scatter.legend()
        {'legend': True, 'position': 'top-right', size: 'large'}
        """
        if legend is not UNDEF:
            try:
                self._legend = legend
                self.update_widget('legend', legend)
                if legend:
                    self.update_widget(
                        'legend_encoding', self.get_legend_encoding()
                    )
            except:
                pass

        if position is not UNDEF:
            try:
                self._legend_position = position
                self.update_widget('legend_position', position)
            except:
                pass

        if size is not UNDEF:
            try:
                self._legend_size = size
                self.update_widget('legend_size', size)
            except:
                pass

        if any_not([legend, position, size], UNDEF):
            return self

        return dict(
            legend = self._legend,
            position = self._legend_position,
            size = self._legend_size,
        )

    def tooltip(
        self,
        enable: Optional[Union[bool, Undefined]] = UNDEF,
        properties: Optional[Union[List[VisualProperty], Undefined]] = UNDEF,
        histograms: Optional[Union[bool, Undefined]] = UNDEF,
        histograms_bins: Optional[Union[int, Dict[str, int], Undefined]] = UNDEF,
        histograms_ranges: Optional[Union[Tuple[float], Dict[str, Tuple[float]], Undefined]] = UNDEF,
        histograms_size: Optional[Union[Size, Undefined]] = UNDEF,
        preview: Optional[Union[str, Undefined]] = UNDEF,
        preview_type: Optional[Union[TooltipPreviewType, Undefined]] = UNDEF,
        preview_text_lines: Optional[Union[int, Undefined]] = UNDEF,
        preview_image_background_color: Optional[Union[Auto, Color, Undefined]] = UNDEF,
        preview_image_position: Optional[Union[TooltipPreviewImagePosition, str, Undefined]] = UNDEF,
        preview_image_size: Optional[Union[TooltipPreviewImageSize, Undefined]] = UNDEF,
        preview_audio_length: Optional[Union[int, Undefined]] = UNDEF,
        preview_audio_loop: Optional[Union[bool, int, Undefined]] = UNDEF,
        preview_audio_controls: Optional[Union[bool, Undefined]] = UNDEF,
        size: Optional[Union[Size, Undefined]] = UNDEF,
    ):
        """
        Set or get the tooltip settings.

        Parameters
        ----------
        enable : bool, optional
            When set to `True`, a tooltip will be shown upon hovering a point.
        properties : list of str, optional
            The visual and data properties that should be shown in the tooltip.
            The visual properties can be some of `x`, `y`, `color`, `opacity`,
            and `size`. Note that visual properties are only shown if they are
            actually used to encode data properties. To reference other data
            properties that are not visually encoded, specify a column of the
            bound DataFrame by its name.
        histograms : bool, optional
            When set to `True`, the tooltip will show histograms of the
            properties
        histograms_bins : int, optional
            The number of bins for all numerical histograms. Or a dictionary of
            property-specific number of bins. Defaults to 20.
        histograms_ranges : (float, float) or dict of (float, float), optional
            The global lower and upper range of all bins. Or a dictionary of
            property-specific lower upper bin ranges. Defaults to
            `(min(), max())`.
        histograms_size : small or medium or large, optional
            The width of the histograms. Must be one of small, medium, or large.
            Defaults to `"small"`.
        preview : str, optional
            A column name of the bound DataFrame that contains preview data.
            Currently three data types are supported: plain text, URL
            referencing an image, and URL referencing an audio file.
        preview_type : str, optional
            The media type of the preview. This can be one of `"text"`,
            `"image"`, or `"audio"`.
        preview_text_lines : int or None, optional
            For text previews, the maximum number of lines that should be
            displayed. Text that exceeds defined limit will be truncated with an
            ellipsis. By default, the line limit is set to `None` to be
            disabled.
        preview_image_background_color : str or None, optional
            For image previews, the background color. By default, the value is
            `None`, which means that image previews' background is transparent.
            If `preview_image_size` is set to `"contain"` and your image does
            not perfectly cover the preview area, you will see the tooltip's
            background color.
        preview_image_position : str, optional
            The image position. This can be one of `"top"`, `"bottom"`,
            `"left"`, `"right"`, or `"center"`. The default value is `"center"`.
        preview_image_size : str, optional
            The size of the image in the context of the preview area. This can
            be one of `"cover"` or `"contain"` and is set to `"contain"` by
            default.
        preview_audio_length : int or None, optional
            The number of seconds that should be played as the audio preview. By
            default (`None`), the audio file is played from the start to the
            end.
        preview_audio_loop : bool, optional
            If `True`, the audio preview is indefinitely looped for the duration
            the tooltip is shown.
        preview_audio_controls : bool, optional
            If `True`, the audio preview will include controls. While you cannot
            interact with the controls (as the tooltip disappears upon leaving a
            point), the controls show the progression and length of the played
            audio.
        size : small or medium or large, optional
            The size of the tooltip. Must be one of `"small"`, `"medium"`, or
            `"large"`. Defaults to `"small"`.

        Returns
        -------
        self or dict
            If no parameters are provided the current tooltip settings are
            returned as a dictionary. Otherwise, `self` is returned.

        Notes
        -----
        See https://developer.mozilla.org/en-US/docs/Web/CSS/background-position
        for details on the behavior of `preview_image_position`.
        See https://developer.mozilla.org/en-US/docs/Web/CSS/background-size
        for details on the behavior of `preview_image_size`.
        See https://developer.mozilla.org/en-US/docs/Web/HTML/Element/audio#loop
        for details on the behavior of `preview_audio_loop`.
        See https://developer.mozilla.org/en-US/docs/Web/HTML/Element/audio#controls
        for details on the behavior of `preview_audio_controls`.

        Examples
        --------
        >>> scatter.tooltip(True)
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(properties=["color", "opacity", "my_column"])
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(histograms=True)
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(histograms_bins=40)
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(histograms_ranges={"my_column": (1, 2)})
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(histograms_size="medium")
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(preview="image_url")
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(preview_type="image")
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(preview_text_lines=3)
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(preview_background_color="red")
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(preview_position="top")
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(preview_size="cover")
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(preview_audio_length=2)
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(preview_audio_loop=True)
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(preview_audio_controls=False)
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip(size="large")
        <jscatter.jscatter.Scatter>

        >>> scatter.tooltip()
        {
          "tooltip": True,
          "properties": ["color", "opacity", "my_column"],
          "histograms": True,
          "histograms_bins": 20,
          "histograms_ranges": {"my_column": (1, 2)},
          "histograms_size": "small",
          "preview": "image_url",
          "preview_type": "image",
          "preview_text_lines": 3,
          "preview_background_color": "red",
          "preview_position": "top",
          "preview_size": "cover",
          "preview_audio_length": 2,
          "preview_audio_loop": True,
          "preview_audio_controls": False,
          "size": "large"
        }
        """
        if enable is not UNDEF:
            self._tooltip = enable
            self.update_widget('tooltip_enable', enable)

        if preview_text_lines is not UNDEF:
            try:
                self._preview_text_lines = max(0, preview_text_lines)
            except TypeError:
                self._preview_text_lines = None
            self.update_widget('tooltip_preview_text_lines', self._preview_text_lines)

        if preview_image_background_color is not UNDEF:
            self._tooltip_preview_image_background_color = preview_image_background_color
            self.update_widget('tooltip_preview_image_background_color', preview_image_background_color)

        if preview_image_position is not UNDEF:
            self._tooltip_preview_image_position = preview_image_position
            self.update_widget('tooltip_preview_image_position', preview_image_position)

        if preview_image_size is not UNDEF:
            self._tooltip_preview_image_size = preview_image_size
            self.update_widget('tooltip_preview_image_size', preview_image_size)

        if preview_audio_length is not UNDEF:
            try:
                self._tooltip_preview_audio_length = max(0, preview_audio_length)
            except TypeError:
                self._tooltip_preview_audio_length = None
            self.update_widget('tooltip_preview_audio_length', self._tooltip_preview_audio_length)

        if preview_audio_loop is not UNDEF:
            self._tooltip_preview_audio_loop = preview_audio_loop
            self.update_widget('tooltip_preview_audio_loop', preview_audio_loop)

        if preview_audio_controls is not UNDEF:
            self._tooltip_preview_audio_controls = preview_audio_controls
            self.update_widget('tooltip_preview_audio_controls', preview_audio_controls)

        if preview_type is not UNDEF:
            self._tooltip_preview_type = preview_type
            self.update_widget('tooltip_preview_type', preview_type)

        if preview is not UNDEF:
            self._tooltip_preview = preview
            self.update_widget('tooltip_preview', preview)

        if properties is not UNDEF:
            self._tooltip_properties = sanitize_tooltip_properties(
                self._data,
                visual_properties,
                properties
            )

            self._tooltip_histograms_bins = {
                property: get_histogram_bins(
                    self._tooltip_histograms_bins,
                    property
                )
                for property
                in self._tooltip_properties
            }

            self._tooltip_histograms_ranges = {
                property: get_histogram_range(
                    self._tooltip_histograms_ranges,
                    property
                )
                for property
                in self._tooltip_properties
            }

        if size is not UNDEF:
            self._tooltip_size = size
            self.update_widget('tooltip_size', size)

        if histograms is not UNDEF:
            self._tooltip_histograms = histograms
            self.update_widget('tooltip_histograms', histograms)

        if histograms_size is not UNDEF:
            self._tooltip_histograms_size = histograms_size
            self.update_widget('tooltip_histograms_size', histograms_size)

        if histograms_bins is not UNDEF:
            self._tooltip_histograms_bins = {
                property: get_histogram_bins(histograms_bins, property)
                for property
                in self._tooltip_properties
            }

        if histograms_ranges is not UNDEF:
            self._tooltip_histograms_ranges = {
                property: get_histogram_range(histograms_ranges, property)
                for property
                in self._tooltip_properties
            }

        if histograms_bins is not UNDEF or histograms_ranges is not UNDEF:
            # Re-create histograms
            self._x_histogram = get_histogram_from_df(
                self._points[:, 0],
                self.get_histogram_bins("x"),
                self.get_histogram_range("x"),
            )
            self.update_widget('x_histogram_range', self.get_histogram_range("x"))
            self.update_widget('x_histogram', self._x_histogram)

            self._y_histogram = get_histogram_from_df(
                self._points[:, 1],
                self.get_histogram_bins("y"),
                self.get_histogram_range("y")
            )
            self.update_widget('y_histogram_range', self.get_histogram_range("y"))
            self.update_widget('y_histogram', self._y_histogram)

            if self._color_by is not None and self._color_categories is None:
                component = self._encodings.data[self._color_by].component
                self._color_histogram = get_histogram_from_df(
                    self._points[:, component],
                    self.get_histogram_bins("color"),
                    self.get_histogram_range("color")
                )
                self.update_widget('color_histogram_range', self.get_histogram_range("color"))
                self.update_widget('color_histogram', self._color_histogram)

            if self._opacity_by is not None and self._opacity_by != "density" and self._opacity_categories is None:
                component = self._encodings.data[self._opacity_by].component
                self._opacity_histogram = get_histogram_from_df(
                    self._points[:, component],
                    self.get_histogram_bins("opacity"),
                    self.get_histogram_range("opacity")
                )
                self.update_widget('opacity_histogram_range', self.get_histogram_range("opacity"))
                self.update_widget('opacity_histogram', self._opacity_histogram)

            if self._size_by is not None and self._size_categories is None:
                component = self._encodings.data[self._size_by].component
                self._size_histogram = get_histogram_from_df(
                    self._points[:, component],
                    self.get_histogram_bins("size"),
                    self.get_histogram_range("size")
                )
                self.update_widget('size_histogram_range', self.get_histogram_range("size"))
                self.update_widget('size_histogram', self._size_histogram)

        if (
            self._tooltip_properties_non_visual is None or
            properties is not UNDEF or
            histograms_bins is not UNDEF
        ):
            self._tooltip_properties_non_visual = get_non_visual_properties(
                self._tooltip_properties
            )

            self._tooltip_properties_non_visual_info = {}
            for property in self._tooltip_properties_non_visual:
                self._tooltip_properties_non_visual_info[property] = dict(
                    scale = get_scale_type_from_df(self._data[property]),
                    domain = get_domain_from_df(self._data[property]),
                    range = self.get_histogram_range(property),
                    histogram = get_histogram_from_df(
                        self._data[property],
                        self.get_histogram_bins(property),
                        self.get_histogram_range(property)
                    ),
                )

            self.update_widget('tooltip_properties_non_visual_info', self._tooltip_properties_non_visual_info)
            self.update_widget('tooltip_properties', self._tooltip_properties)

        if any_not([enable, properties, size, histograms, histograms_bins, histograms_ranges, histograms_size], UNDEF):
            return self

        return dict(
            enable = self._tooltip,
            properties = self._tooltip_properties,
            histograms = self._tooltip_histograms,
            histograms_bins = self._tooltip_histograms_bins,
            histograms_ranges = self._tooltip_histograms_ranges,
            histograms_size = self._tooltip_histograms_size,
            preview = self._tooltip_preview,
            preview_type = self._tooltip_preview_type,
            preview_text_lines = self._tooltip_preview_text_lines,
            preview_image_background_color = self._tooltip_preview_image_background_color,
            preview_image_position = self._tooltip_preview_image_position,
            preview_image_size = self._tooltip_preview_image_size,
            preview_audio_length = self._tooltip_preview_audio_length,
            preview_audio_loop = self._tooltip_preview_audio_loop,
            preview_audio_controls = self._tooltip_preview_audio_controls,
            size = self._tooltip_size,
        )

    def annotations(
        self,
        annotations: Optional[Union[List[Union[Line, HLine, VLine, Rect]], Undefined]] = UNDEF,
    ):
        """
        Draw line-based annotatons

        Parameters
        ----------
        annotations : list of `Line`, `HLine`, `VLine`, `Rect`, optional
            A list of annotations (`Line`, `HLine`, `VLine`, `Rect`) or `None`.
            When set to `None` or `[]` all annotations are removed.

        Returns
        -------
        self or dict
            If no parameters are provided the current annotation settings are
            returned as a dictionary. Otherwise, `self` is returned.

        Examples
        --------
        >>> scatter.annotations([HLine(0), VLine(0)])
        <jscatter.jscatter.Scatter>

        >>> scatter.annotations()
        {'annotations': [HLine(0), VLine(0)]}
        """

        if annotations is not UNDEF:
            self._annotations = annotations
            self.update_widget(
                'annotations',
                normalize_annotations(
                    self._annotations,
                    self._x_scale,
                    self._y_scale
                )
            )

        if any_not([annotations], UNDEF):
            return self

        return dict(
            annotations = self._annotations
        )

    def zoom(
        self,
        to: Optional[Union[List[int], np.ndarray, None, Undefined]] = UNDEF,
        animation: Optional[Union[bool, int, Undefined]] = UNDEF,
        padding: Optional[Union[float, Undefined]] = UNDEF,
        on_selection: Optional[Union[bool, Undefined]] = UNDEF,
        on_filter: Optional[Union[bool, Undefined]] = UNDEF,
    ):
        """
        Zoom to a set of points.

        Parameters
        ----------
        to : array_like, optional
            A list of point indices or `None`. When set to `None` the
            camera's zoom state is reset.
        animation : bool or int, optional
            Whether to animate the transitioning of the camera from the current
            to the new zoom state. This can either be a Boolean value or an
            integer specifying the duration of the animation in milliseconds.
        padding : float, optional
            Relative padding around the bounding box of the target points. Zero
            stands for no padding at all and one stands for a padding that is as
            wide and tall as the width and height of the points' bounding box.
        on_selection : bool, optional
            If `True` jscatter will automatically zoom to selected points.
        on_filter : bool, optional
            If `True` jscatter will automatically zoom to filtered points.

        Returns
        -------
        self or dict
            If no parameters are provided the current zoom target is
            returned as a dictionary. Otherwise, `self` is returned.

        Examples
        --------
        >>> scatter.zoom(target=[0, 1, 2])
        <jscatter.jscatter.Scatter>

        >>> scatter.zoom(target=scatter.selection(), animation=True)
        <jscatter.jscatter.Scatter>

        >>> scatter.zoom(target=None, animation=500, padding=0)
        <jscatter.jscatter.Scatter>

        >>> scatter.zoom()
        {'target': None, 'animation': 1000, 'padding': 0.333, 'on_selection': False, 'on_filter': False}
        """

        if animation is not UNDEF:
            if isinstance(animation, bool):
                self._zoom_animation = animation = 1000 if animation else 0
            else:
                self._zoom_animation = animation

            self.update_widget('zoom_animation', self._zoom_animation)

        if padding is not UNDEF:
            self._zoom_padding = padding
            self.update_widget('zoom_padding', padding)

        if on_selection is not UNDEF:
            self._zoom_on_selection = on_selection
            self.update_widget('zoom_on_selection', on_selection)

        if on_filter is not UNDEF:
            self._zoom_on_filter = on_filter
            self.update_widget('zoom_on_filter', on_filter)

        if to is not UNDEF:
            self._zoom_to = to
            self.update_widget('zoom_to', to)
            self._zoom_to_call_idx += 1
            self.update_widget('zoom_to_call_idx', self._zoom_to_call_idx)

        if any_not([to, animation, padding, on_selection], UNDEF):
            return self

        return dict(
            to = self._zoom_to,
            animation = self._zoom_animation,
            padding = self._zoom_padding,
            on_selection = self._zoom_on_selection,
            on_filter = self._zoom_on_filter,
        )



    def options(
        self,
        transition_points: Optional[Union[bool, Undefined]] = UNDEF,
        transition_points_duration: Optional[Union[int, Undefined]] = UNDEF,
        regl_scatterplot_options: Optional[Union[dict, Undefined]] = UNDEF
    ):
        """
        Set or get additional options.

        Parameters
        ----------
        transition_points : bool, optional
            Whether to enable or disable the potential animated transitioning of
            points as their coordinates update. If `False`, points will never be
            animated.
        transition_points_duration : int, optional
            The time of the animated point transition in milliseconds. The
            default value is `3000`.
        regl_scatterplot_options : dict, optional
            A dictionary for specifying any kinds of custom Regl-Scatterplot
            options.

        Returns
        -------
        self or dict
            If no parameters are provided the current options are returned as a
            dictionary. Otherwise, `self` is returned.

        Notes
        -----
        The scatter rendering is done with a JavaScript library called
        regl-scatterplot. Please see the following page for available settings:
        https://github.com/flekschas/regl-scatterplot

        Examples
        --------
        >>> scatter.options(transition_points=True)
        <jscatter.jscatter.Scatter>

        >>> scatter.options(transition_points_duration=500)
        <jscatter.jscatter.Scatter>

        >>> scatter.options(regl_scatterplot_options=dict(deselectOnEscape=False))
        <jscatter.jscatter.Scatter>

        >>> scatter.options()
        {transition_points=True, transition_points_duration=500, regl_scatterplot_options={deselectOnEscape=False}}
        """
        if transition_points is not UNDEF:
            self._transition_points = bool(transition_points)

        if transition_points_duration is not UNDEF:
            try:
                self._transition_points_duration = int(transition_points_duration)
                assert self._transition_points_duration >= 0
            except:
                self._transition_points_duration = DEFAULT_TRANSITION_POINTS_DURATION

            self.update_widget('transition_points_duration', self._transition_points_duration)

        if regl_scatterplot_options is not UNDEF:
            try:
                self._regl_scatterplot_options = regl_scatterplot_options
                self.update_widget('regl_scatterplot_options', regl_scatterplot_options)
            except:
                pass

            return self

        if any_not([transition_points, transition_points_duration, regl_scatterplot_options], UNDEF):
            return self

        return dict(
            transition_points=self._transition_points,
            transition_points_duration=self._transition_points_duration,
            regl_scatterplot_options=self._regl_scatterplot_options
        )

    def pixels(self):
        """
        Get the current view as an image

        Returns
        -------
        np.ndarray or None
            Once the pixels have been downloaded from the JavaScript kernel,
            they are returned as a Numpy array. Otherwise, `None` is returned.

        Notes
        -----
        In order to retrieve the current view as an image, you first have to
        download the pixels from the JavaScript into the Python kernel. You
        can do this by clicking on the widget button that contains a camera
        icon.

        Examples
        --------
        >>> scatter.pixels()
        array([[[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                ...,
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]]], dtype=uint8)
        """
        if self._widget is not None:
            assert self._widget.view_data is not None and len(self._widget.view_data) > 0, 'Download pixels first by clicking on the button with the camera icon.'
            assert self._widget.view_shape is not None and len(self._widget.view_shape) == 2, 'Download pixels first by clicking on the button with the camera icon.'

            self._pixels = np.asarray(self._widget.view_data).astype(np.uint8)
            self._pixels = self._pixels.reshape([*self._widget.view_shape, 4])

        return self._pixels

    @property
    def _data_index(self):
        if self._data is None:
            return np.arange(self._n)
        return self._data.index

    @property
    def _color_data_dimension(self):
        if self._color_by is None:
            return None
        return f'{self._color_by}:{to_scale_type(self._color_norm)}'

    @property
    def _opacity_data_dimension(self):
        if self._opacity_by is None or self._opacity_by == 'density':
            return None
        return f'{self._opacity_by}:{to_scale_type(self._opacity_norm)}'

    @property
    def _size_data_dimension(self):
        if self._size_by is None:
            return None
        return f'{self._size_by}:{to_scale_type(self._size_norm)}'

    @property
    def _connection_color_data_dimension(self):
        if self._connection_color_by is None or self._connection_color_by == 'segment':
            return None
        return f'{self._connection_color_by}:{to_scale_type(self._connection_color_norm)}'

    @property
    def _connection_opacity_data_dimension(self):
        if self._connection_opacity_by is None:
            return None
        return f'{self._connection_opacity_by}:{to_scale_type(self._connection_opacity_norm)}'

    @property
    def _connection_size_data_dimension(self):
        if self._connection_size_by is None:
            return None
        return f'{self._connection_size_by}:{to_scale_type(self._connection_size_norm)}'

    @property
    def js_color_by(self):
        if self._color_data_dimension is not None:
            return component_idx_to_name(
                self._encodings.data[self._color_data_dimension].component
            )

        return None

    @property
    def js_opacity_by(self):
        if self._opacity_by == 'density':
            return 'density'

        if self._opacity_data_dimension is not None:
            return component_idx_to_name(
                self._encodings.data[self._opacity_data_dimension].component
            )

        return None

    @property
    def js_size_by(self):
        if self._size_data_dimension is not None:
            return component_idx_to_name(
                self._encodings.data[self._size_data_dimension].component
            )

        return None

    @property
    def js_connection_color_by(self):
        if self._connection_color_by == 'segment':
            return 'segment'

        if self._connection_color == 'inherit' or self._connection_color_by == 'inherit':
            return 'inherit'

        if self._connection_color_data_dimension is not None:
            return component_idx_to_name(
                self._encodings.data[self._connection_color_data_dimension].component
            )

        return None

    @property
    def js_connection_opacity_by(self):
        if self._connection_opacity_data_dimension is not None:
            return component_idx_to_name(
                self._encodings.data[self._connection_opacity_data_dimension].component
            )

        return None

    @property
    def js_connection_size_by(self):
        if self._connection_size_data_dimension is not None:
            return component_idx_to_name(
                self._encodings.data[self._connection_size_data_dimension].component
            )

        return None

    @property
    def widget(self):
        if self._widget is not None:
            return self._widget

        self._widget = JupyterScatter(
            data=self._data,
            annotations=normalize_annotations(
                self._annotations,
                self._x_scale,
                self._y_scale
            ),
            axes=self._axes,
            axes_color=self.get_axes_color(),
            axes_grid=self._axes_grid,
            axes_labels=self.get_axes_labels(),
            background_color=self._background_color,
            background_image=self._background_image,
            camera_distance=self._camera_distance,
            camera_rotation=self._camera_rotation,
            camera_target=self._camera_target,
            camera_view=self._camera_view,
            color=order_map(self._color_map, self._color_order) if self._color_map else self._color,
            color_by=self.js_color_by,
            color_domain=get_domain(self, 'color'),
            color_histogram=self._color_histogram,
            color_histogram_range=self.get_histogram_range("color"),
            color_hover=self._color_hover,
            color_scale=get_scale(self, 'color'),
            color_selected=self._color_selected,
            color_title=self._color_by,
            connect=bool(self._connect_by),
            connection_color=order_map(self._connection_color_map, self._connection_color_order) if self._connection_color_map else self._connection_color,
            connection_color_by=self.js_connection_color_by,
            connection_color_hover=self._connection_color_hover,
            connection_color_selected=self._connection_color_selected,
            connection_opacity=order_map(self._connection_opacity_map, self._connection_opacity_order) if self._connection_opacity_map else self._connection_opacity,
            connection_opacity_by=self.js_connection_opacity_by,
            connection_size=order_map(self._connection_size_map, self._connection_size_order) if self._connection_size_map else self._connection_size,
            connection_size_by=self.js_connection_size_by,
            filter=self._filtered_points_idxs,
            height=self._height,
            lasso_color=self._lasso_color,
            lasso_initiator=self._lasso_initiator,
            lasso_min_delay=self._lasso_min_delay,
            lasso_min_dist=self._lasso_min_dist,
            lasso_on_long_press=self._lasso_on_long_press,
            legend=self._legend,
            legend_color=self.get_legend_color(),
            legend_encoding=self.get_legend_encoding(),
            legend_position=self._legend_position,
            legend_size=self._legend_size,
            mouse_mode=self._mouse_mode,
            opacity=order_map(self._opacity_map, self._opacity_order) if self._opacity_map else self._opacity,
            opacity_by=self.js_opacity_by,
            opacity_domain=get_domain(self, 'opacity'),
            opacity_histogram=self._opacity_histogram,
            opacity_histogram_range=self.get_histogram_range("opacity"),
            opacity_scale=get_scale(self, 'opacity'),
            opacity_title=self._opacity_by,
            opacity_unselected=self._opacity_unselected,
            points=self.get_point_list(),
            reticle=self._reticle,
            reticle_color=self.get_reticle_color(),
            selection=self._selected_points_idxs,
            size=order_map(self._size_map, self._size_order) if self._size_map else self._size,
            size_by=self.js_size_by,
            size_domain=get_domain(self, 'size'),
            size_histogram=self._size_histogram,
            size_histogram_range=self.get_histogram_range("size"),
            size_scale=get_scale(self, 'size'),
            size_title=self._size_by,
            tooltip_enable=self._tooltip,
            tooltip_color=self.get_tooltip_color(),
            tooltip_properties=self._tooltip_properties,
            tooltip_properties_non_visual_info=self._tooltip_properties_non_visual_info,
            tooltip_histograms=self._tooltip_histograms,
            tooltip_histograms_ranges=self._tooltip_histograms_ranges,
            tooltip_histograms_size=self._tooltip_histograms_size,
            tooltip_preview=self._tooltip_preview,
            tooltip_preview_type=self._tooltip_preview_type,
            tooltip_preview_text_lines=self._tooltip_preview_text_lines,
            tooltip_preview_image_background_color=self._tooltip_preview_image_background_color,
            tooltip_preview_image_position=self._tooltip_preview_image_position,
            tooltip_preview_image_size=self._tooltip_preview_image_size,
            tooltip_preview_audio_length=self._tooltip_preview_audio_length,
            tooltip_preview_audio_loop=self._tooltip_preview_audio_loop,
            tooltip_preview_audio_controls=self._tooltip_preview_audio_controls,
            tooltip_size=self._tooltip_size,
            width=self._width,
            x_domain=self._x_domain,
            x_histogram=self._x_histogram,
            x_histogram_range=self.get_histogram_range("x"),
            x_scale=to_scale_type(self._x_scale),
            x_scale_domain=self._x_scale_domain,
            x_title=self._x_by,
            y_domain=self._y_domain,
            y_histogram=self._y_histogram,
            y_histogram_range=self.get_histogram_range("y"),
            y_scale=to_scale_type(self._y_scale),
            y_scale_domain=self._y_scale_domain,
            y_title=self._y_by,
            zoom_animation=self._zoom_animation,
            zoom_on_filter=self._zoom_on_filter,
            zoom_on_selection=self._zoom_on_selection,
            zoom_padding=self._zoom_padding,
            zoom_to=self._zoom_to,
            regl_scatterplot_options=self._regl_scatterplot_options,
            transition_points_duration=self._transition_points_duration,
        )

        return self._widget

    def update_widget(self, prop, val):
        if self._widget is not None:
            setattr(self._widget, prop, val)

    def get_histogram_bins(self, property):
        return get_histogram_bins(self._tooltip_histograms_bins, property)

    def get_histogram_range(self, property):
        return get_histogram_range(self._tooltip_histograms_ranges, property)

    def get_axes_labels(self):
        if self._axes_labels == True:
            if self._data is None:
                return ['x', 'y']
            else:
                return [self._x_by, self._y_by]

        return self._axes_labels

    def show(self):
        """
        Show the scatter plot widget

        Returns
        -------
        widget
            The widget that is being rendering by Jupyter

        Examples
        --------
        >>> scatter.show()
        """
        return self.widget.show()

    def show_tooltip(self, pointIdx: int):
        """
        Programmatically show the tooltip for a point.

        If the widget has not been instantiated yet or the tooltip is not
        activated, this method is a noop.

        Returns
        -------
        self
            The Scatter instance itself is returned.

        Examples
        --------
        >>> scatter.show_tooltip(42)
        """

        if self._widget is not None and self._tooltip == True:
            self._widget.show_tooltip(pointIdx)

        return self


def plot(
    x: Union[str, List[float], np.ndarray],
    y: Union[str, List[float], np.ndarray],
    data: Optional[pd.DataFrame] = None,
    **kwargs
):
    """
    Create a scatter instance and immediately show it as a widget.

    Parameters
    ----------
    x : str, array_like
        The x coordinates given as either an array-like list of coordinates
        or a string referencing a column in `data`.
    y : str, array_like
        The y coordinates given as either an array-like list of coordinates
        or a string referencing a column in `data`.
    data : pd.DataFrame, optional
        The data frame that holds the x and y coordinates as well as other
        possible dimensions that can be used for color, size, or opacity
        encoding.
    kwargs : optional
        Options to customize the scatter instance and the visual encoding.
        See https://jupyter-scatter.dev/api#properties
        for a complete list of all properties.

    Returns
    -------
    widget
        The scatter plot widget

    See Also
    --------
    Scatter : Create a scatter instance.
    show : Show the scatter plot widget

    Examples
    --------
    >>> plot(x=np.arange(5), y=np.arange(5))
    >>> plot(data=df, x='weight', y='speed', color_by='length')
    """
    return Scatter(x, y, data, **kwargs).show()
