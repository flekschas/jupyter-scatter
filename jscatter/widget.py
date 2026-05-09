import pathlib
import typing as t

import anywidget
import numpy as np
import pandas as pd
from traitlets import Bool, Dict, Enum, Float, Int, List, Unicode, Union, observe

from .annotations_traits import (
    HLine,
    Line,
    Rect,
    VLine,
)
from .annotations_traits import (
    serialization as annotation_serialization,
)
from .label_placement import DEFAULT_TILE_SIZE, INITIAL_TILE, LabelPlacement, to_js
from .serializers import df_serialization, ndarray_serialization
from .traittypes import Array, DataFrame
from .types import UNDEF, Undefined
from .utils import is_categorical_data

SELECTION_DTYPE = 'uint32'
EVENT_TYPES = {
    'TOOLTIP': 'tooltip',
    'VIEW_RESET': 'view_reset',
    'VIEW_SAVE': 'view_save',
}
BRUSH_SIZE_MIN = 1
BRUSH_SIZE_MAX = 128
TOOLBAR_BUTTONS_DEFAULT = [
    'pan_zoom',
    'lasso',
    'lasso_type',
    'lasso_brush_size',
    'divider',
    'full_screen',
    'download',
    'reset',
]
TOOLBAR_BUTTONS_VALID = {
    *TOOLBAR_BUTTONS_DEFAULT,
    'save',
}


def component_idx_to_name(idx):
    if idx == 2:
        return 'valueA'

    if idx == 3:
        return 'valueB'

    return None


def sorting_to_dict(sorting):
    out = dict()
    for order_idx, original_idx in enumerate(sorting):
        out[original_idx] = order_idx
    return out


def get_record(data, index):
    out = data.iloc[index].copy()
    fill_na = {c: 'NA' for c in data.columns if is_categorical_data(data[c])}
    return out.fillna(fill_na)


class JupyterScatter(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / 'bundle.js'
    _css = pathlib.Path(__file__).parent / 'bundle.css'

    data: t.Optional[pd.DataFrame]
    label_placement: t.Optional[LabelPlacement]

    # For debugging
    dom_element_id = Unicode(read_only=True).tag(sync=True)

    # Data
    points = Array(default_value=None).tag(sync=True, **ndarray_serialization)
    transition_points = Bool(False).tag(sync=True)
    transition_points_duration = Int(3000).tag(sync=True)
    prevent_filter_reset = Bool(False).tag(sync=True)
    non_spatial_points_update = Bool(False).tag(sync=True)
    selection = Array(default_value=None, allow_none=True).tag(
        sync=True, **ndarray_serialization
    )
    filter = Array(default_value=None, allow_none=True).tag(
        sync=True, **ndarray_serialization
    )
    hovering = Int(None, allow_none=True).tag(sync=True)
    point_order = Array(default_value=None, allow_none=True).tag(
        sync=True, **ndarray_serialization
    )

    # Channel titles
    x_title = Unicode(None, allow_none=True).tag(sync=True)
    y_title = Unicode(None, allow_none=True).tag(sync=True)
    color_title = Unicode(None, allow_none=True).tag(sync=True)
    opacity_title = Unicode(None, allow_none=True).tag(sync=True)
    size_title = Unicode(None, allow_none=True).tag(sync=True)

    # Scales
    x_scale = Unicode(None, allow_none=True).tag(sync=True)
    y_scale = Unicode(None, allow_none=True).tag(sync=True)
    color_scale = Unicode(None, allow_none=True).tag(sync=True)
    opacity_scale = Unicode(None, allow_none=True).tag(sync=True)
    size_scale = Unicode(None, allow_none=True).tag(sync=True)

    # Domains
    x_domain = List(minlen=2, maxlen=2).tag(sync=True)
    y_domain = List(minlen=2, maxlen=2).tag(sync=True)
    color_domain = Union([Dict(), List(minlen=2, maxlen=2)], allow_none=True).tag(
        sync=True
    )
    opacity_domain = Union([Dict(), List(minlen=2, maxlen=2)], allow_none=True).tag(
        sync=True
    )
    size_domain = Union([Dict(), List(minlen=2, maxlen=2)], allow_none=True).tag(
        sync=True
    )

    # Scale domains
    x_scale_domain = List(minlen=2, maxlen=2).tag(sync=True)
    y_scale_domain = List(minlen=2, maxlen=2).tag(sync=True)

    # Histograms
    x_histogram = List(None, allow_none=True).tag(sync=True)
    y_histogram = List(None, allow_none=True).tag(sync=True)
    color_histogram = List(None, allow_none=True).tag(sync=True)
    opacity_histogram = List(None, allow_none=True).tag(sync=True)
    size_histogram = List(None, allow_none=True).tag(sync=True)

    # Histogram ranges
    x_histogram_range = List(None, allow_none=True, minlen=2, maxlen=2).tag(sync=True)
    y_histogram_range = List(None, allow_none=True, minlen=2, maxlen=2).tag(sync=True)
    color_histogram_range = List(None, allow_none=True, minlen=2, maxlen=2).tag(
        sync=True
    )
    opacity_histogram_range = List(None, allow_none=True, minlen=2, maxlen=2).tag(
        sync=True
    )
    size_histogram_range = List(None, allow_none=True, minlen=2, maxlen=2).tag(
        sync=True
    )

    # Annotations
    annotations = List(
        trait=Union([Line(), HLine(), VLine(), Rect()]),
        default_value=None,
        allow_none=True,
    ).tag(sync=True, **annotation_serialization)

    # Labels
    labels = DataFrame(default_value=None, allow_none=True).tag(
        sync=True, **df_serialization
    )
    label_shadow_color = Unicode(None, allow_none=True).tag(sync=True)
    label_align = Enum(
        [
            'top',
            'top-right',
            'top-left',
            'bottom',
            'bottom-right',
            'bottom-left',
            'left',
            'right',
            'center',
        ],
        default_value='center',
    ).tag(sync=True)
    label_offset = List([0, 0], minlen=2, maxlen=2).tag(sync=True)
    label_scale_function = Enum(
        [
            'asinh',
            'constant',
        ],
        default_value='asinh',
    ).tag(sync=True)
    label_tiles = List(
        trait=Unicode(), default_value=[INITIAL_TILE], read_only=True
    ).tag(sync=True)
    label_tile_size = Int(DEFAULT_TILE_SIZE).tag(sync=True)

    # View properties
    camera_target = List([0, 0]).tag(sync=True)
    camera_distance = Float(1).tag(sync=True)
    camera_rotation = Float(1).tag(sync=True)
    camera_view = List(None, allow_none=True).tag(sync=True)
    camera_is_fixed = Bool(False).tag(sync=True)

    # Zoom properties
    zoom_to = Array(default_value=None, allow_none=True).tag(
        sync=True, **ndarray_serialization
    )
    zoom_to_call_idx = Int(0).tag(sync=True)
    zoom_animation = Int(1000).tag(sync=True)
    zoom_padding = Float(0.333).tag(sync=True)
    zoom_on_selection = Bool(False).tag(sync=True)
    zoom_on_filter = Bool(False).tag(sync=True)
    zoom_level = Float(0.0, read_only=True).tag(sync=True)

    # Interaction properties
    mouse_mode = Enum(['panZoom', 'lasso', 'rotate'], default_value='panZoom').tag(
        sync=True
    )
    lasso_type = Enum(['freeform', 'brush', 'rectangle'], default_value='freeform').tag(
        sync=True
    )
    toolbar_buttons = List(
        Unicode(),
        default_value=TOOLBAR_BUTTONS_DEFAULT,
    ).tag(sync=True)
    lasso_initiator = Bool().tag(sync=True)
    lasso_on_long_press = Bool().tag(sync=True)
    lasso_selection_polygon = Array(
        default_value=None,
        allow_none=True,
        read_only=True,
    ).tag(sync=True, **ndarray_serialization)
    lasso_brush_size = Int(24, min=BRUSH_SIZE_MIN, max=BRUSH_SIZE_MAX).tag(sync=True)

    # Axes
    axes = Bool().tag(sync=True)
    axes_grid = Bool().tag(sync=True)
    axes_color = List(default_value=[0, 0, 0, 1], minlen=4, maxlen=4).tag(sync=True)
    axes_labels = Union([Bool(), List(minlen=1, maxlen=2)]).tag(sync=True)

    # Legend
    legend = Bool().tag(sync=True)
    legend_position = Enum(
        [
            'top',
            'top-right',
            'top-left',
            'bottom',
            'bottom-right',
            'bottom-left',
            'left',
            'right',
            'center',
        ],
        default_value='top-left',
    ).tag(sync=True)
    legend_size = Enum(['small', 'medium', 'large'], default_value='small').tag(
        sync=True
    )
    legend_color = List(default_value=[0, 0, 0, 1], minlen=4, maxlen=4).tag(sync=True)
    legend_encoding = Dict(dict()).tag(sync=True)

    # Tooltip
    tooltip_enable = Bool().tag(sync=True)
    tooltip_size = Enum(['small', 'medium', 'large'], default_value='small').tag(
        sync=True
    )
    tooltip_color = List(default_value=[0, 0, 0, 1], minlen=4, maxlen=4).tag(sync=True)
    tooltip_properties = List(default_value=['x', 'y', 'color', 'opacity', 'size']).tag(
        sync=True
    )
    tooltip_properties_non_visual_info = Dict(dict()).tag(sync=True)
    tooltip_histograms = Union([Bool(), List()]).tag(sync=True)
    tooltip_histograms_ranges = Dict(dict()).tag(sync=True)
    tooltip_histograms_size = Enum(
        ['small', 'medium', 'large'], default_value='small'
    ).tag(sync=True)
    tooltip_preview = Unicode(None, allow_none=True).tag(sync=True)
    tooltip_preview_type = Enum(['text', 'image', 'audio'], default_value='text').tag(
        sync=True
    )
    tooltip_preview_text_lines = Int(default_value=3, allow_none=True).tag(sync=True)
    tooltip_preview_image_background_color = Union(
        [Enum(['auto']), Unicode()], default_value='auto'
    ).tag(sync=True)
    tooltip_preview_image_position = Union(
        [Enum(['top', 'left', 'right', 'bottom', 'center']), Unicode()],
        allow_none=True,
        default_value=None,
    ).tag(sync=True)
    tooltip_preview_image_size = Enum(
        ['contain', 'cover'], allow_none=True, default_value=None
    ).tag(sync=True)
    tooltip_preview_image_height = Int(None, allow_none=True).tag(sync=True)
    tooltip_preview_audio_length = Int(None, allow_none=True).tag(sync=True)
    tooltip_preview_audio_loop = Bool().tag(sync=True)
    tooltip_preview_audio_controls = Bool().tag(sync=True)

    # Options
    color = Union(
        [
            Union([Unicode(), List(minlen=4, maxlen=4)]),
            List(Union([Unicode(), List(minlen=4, maxlen=4)])),
        ]
    ).tag(sync=True)
    color_selected = Union(
        [
            Union([Unicode(), List(minlen=4, maxlen=4)]),
            List(Union([Unicode(), List(minlen=4, maxlen=4)])),
        ]
    ).tag(sync=True)
    color_hover = Union(
        [
            Union([Unicode(), List(minlen=4, maxlen=4)]),
            List(Union([Unicode(), List(minlen=4, maxlen=4)])),
        ]
    ).tag(sync=True)
    color_by = Enum(
        [None, 'valueA', 'valueB'], allow_none=True, default_value=None
    ).tag(sync=True)
    opacity = Union([Float(), List(Float())], allow_none=True).tag(sync=True)
    opacity_unselected = Float().tag(sync=True)
    opacity_by = Enum(
        [None, 'valueA', 'valueB', 'density'], allow_none=True, default_value=None
    ).tag(sync=True)
    size = Union([Union([Int(), Float()]), List(Union([Int(), Float()]))]).tag(
        sync=True
    )
    size_by = Enum([None, 'valueA', 'valueB'], allow_none=True, default_value=None).tag(
        sync=True
    )
    size_scale_function = Enum(
        ['asinh', 'constant', 'linear'], default_value='asinh'
    ).tag(sync=True)
    connect = Bool().tag(sync=True)
    connection_color = Union(
        [
            Union([Unicode(), List(minlen=4, maxlen=4)]),
            List(Union([Unicode(), List(minlen=4, maxlen=4)])),
        ]
    ).tag(sync=True)
    connection_color_selected = Union(
        [
            Union([Unicode(), List(minlen=4, maxlen=4)]),
            List(Union([Unicode(), List(minlen=4, maxlen=4)])),
        ]
    ).tag(sync=True)
    connection_color_hover = Union(
        [
            Union([Unicode(), List(minlen=4, maxlen=4)]),
            List(Union([Unicode(), List(minlen=4, maxlen=4)])),
        ]
    ).tag(sync=True)
    connection_color_by = Enum(
        [None, 'valueA', 'valueB', 'segment', 'inherit'],
        allow_none=True,
        default_value=None,
    ).tag(sync=True)
    connection_opacity = Union([Float(), List(Float())], allow_none=True).tag(sync=True)
    connection_opacity_by = Enum(
        [None, 'valueA', 'valueB', 'segment'], allow_none=True, default_value=None
    ).tag(sync=True)
    connection_size = Union(
        [Union([Int(), Float()]), List(Union([Int(), Float()]))]
    ).tag(sync=True)
    connection_size_by = Enum(
        [None, 'valueA', 'valueB', 'segment'], allow_none=True, default_value=None
    ).tag(sync=True)
    width = Union([Unicode(), Int()], default_value='auto').tag(sync=True)
    height = Int().tag(sync=True)
    background_color = Union([Unicode(), List(minlen=4, maxlen=4)]).tag(sync=True)
    background_image = Unicode(None, allow_none=True).tag(sync=True)
    lasso_color = Union([Unicode(), List(minlen=4, maxlen=4)]).tag(sync=True)
    lasso_min_delay = Int().tag(sync=True)
    lasso_min_dist = Float().tag(sync=True)
    # selection_outline_width = Int().tag(sync=True)
    reticle = Bool().tag(sync=True)
    reticle_color = Union([Unicode(), List(minlen=4, maxlen=4)]).tag(sync=True)

    regl_scatterplot_options = Dict(dict()).tag(sync=True)

    view_download = Unicode(None, allow_none=True).tag(
        sync=True
    )  # Used for triggering a download
    view_data = Array(default_value=None, allow_none=True, read_only=True).tag(
        sync=True, **ndarray_serialization
    )

    # For synchronyzing view changes across scatter plot instances
    view_sync = Unicode(None, allow_none=True).tag(sync=True)

    event_types = Dict(EVENT_TYPES, read_only=True).tag(sync=True)

    def __init__(
        self,
        data: t.Optional[pd.DataFrame],
        label_placement: t.Optional[LabelPlacement] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.data = data
        self.label_placement = label_placement
        self.on_msg(self._handle_custom_msg)

    def _compare(self, a, b):
        """Compare two values for equality."""
        if self._is_numpy(a) or self._is_numpy(b):
            return np.array_equal(a, b)

        if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
            return a.equals(b)

        if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
            return False

        return a == b

    def _handle_custom_msg(self, event: dict, buffers):
        if event['type'] == EVENT_TYPES['TOOLTIP']:
            self._handle_tooltip(event)

    def _handle_tooltip(self, event: dict):
        if isinstance(self.data, pd.DataFrame):
            data = get_record(self.data, event['index'])
            self.send(
                {
                    'type': EVENT_TYPES['TOOLTIP'],
                    'index': event['index'],
                    'preview': data[event['preview']]
                    if event['preview'] is not None
                    else None,
                    'properties': data[event['properties']].to_dict(),
                }
            )

    def show_tooltip(self, point_idx):
        data = get_record(self.data, point_idx)
        self.send(
            {
                'type': EVENT_TYPES['TOOLTIP'],
                'show': True,
                'index': point_idx,
                'preview': data[self.tooltip_preview]
                if self.tooltip_preview is not None
                else None,
                'properties': data[
                    self.tooltip_properties_non_visual_info.keys()
                ].to_dict()
                if self.tooltip_properties_non_visual_info is not None
                else {},
            }
        )

    @observe('label_tiles')
    def _label_tiles_changed(self, change):
        if self.label_placement is None:
            self.labels = None
        else:
            self.labels = to_js(
                self.label_placement.get_labels_from_tiles(change['new'])
            )

    def reset_view(self, animation: int = 0, data_extent: bool = False):
        if data_extent:
            self.send(
                {
                    'type': EVENT_TYPES['VIEW_RESET'],
                    'area': {
                        'x': self.points[:, 0].min(),
                        'width': self.points[:, 0].max() - self.points[:, 0].min(),
                        'y': self.points[:, 1].min(),
                        'height': self.points[:, 1].max() - self.points[:, 1].min(),
                    },
                    'animation': animation,
                }
            )
        else:
            self.send({'type': EVENT_TYPES['VIEW_RESET'], 'animation': animation})

    def show(self, buttons=UNDEF):
        if buttons is not UNDEF:
            unknown = set(buttons) - TOOLBAR_BUTTONS_VALID
            if unknown:
                import warnings

                warnings.warn(
                    f'Unknown toolbar button(s): {unknown}. '
                    f'Valid options: {sorted(TOOLBAR_BUTTONS_VALID)}',
                    stacklevel=2,
                )
            self.toolbar_buttons = [b for b in buttons if b in TOOLBAR_BUTTONS_VALID]
        else:
            self.toolbar_buttons = list(TOOLBAR_BUTTONS_DEFAULT)
        return self
