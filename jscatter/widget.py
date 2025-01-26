import ipywidgets as widgets
import numpy as np
import anywidget
import pandas as pd
import pathlib
import typing as t

from traitlets import Bool, Dict, Enum, Float, Int, List, Unicode, Union
from traittypes import Array

from .annotations_traits import (
    Line,
    HLine,
    VLine,
    Rect,
    serialization as annotation_serialization,
)
from .types import UNDEF, Undefined, WidgetButtons
from .widgets import Button, ButtonChoice, ButtonIntSlider, Divider

SELECTION_DTYPE = 'uint32'
EVENT_TYPES = {
    'FULL_SCREEN': 'full_screen',
    'TOOLTIP': 'tooltip',
    'VIEW_DOWNLOAD': 'view_download',
    'VIEW_RESET': 'view_reset',
    'VIEW_SAVE': 'view_save',
}
BRUSH_SIZE_MIN = 1
BRUSH_SIZE_MAX = 128

divider = Divider()


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


# Code extracted from maartenbreddels ipyvolume
def array_to_binary(ar, obj=None, force_contiguous=True):
    if ar is None:
        return None
    if ar.dtype.kind not in ['u', 'i', 'f']:  # ints and floats
        raise ValueError('unsupported dtype: %s' % (ar.dtype))
    if ar.dtype == np.float64:  # WebGL does not support float64, case it here
        ar = ar.astype(np.float32)
    if ar.dtype == np.int64:  # JS does not support int64
        ar = ar.astype(np.int32)
    if force_contiguous and not ar.flags['C_CONTIGUOUS']:  # make sure it's contiguous
        ar = np.ascontiguousarray(ar)
    return {'view': memoryview(ar), 'dtype': str(ar.dtype), 'shape': ar.shape}


def binary_to_array(value, obj=None):
    if value is None:
        return None
    return np.frombuffer(value['view'], dtype=value['dtype']).reshape(value['shape'])


ndarray_serialization = dict(to_json=array_to_binary, from_json=binary_to_array)


class JupyterScatter(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / 'bundle.js'

    # For debugging
    dom_element_id = Unicode(read_only=True).tag(sync=True)

    # Data
    points = Array(default_value=None).tag(sync=True, **ndarray_serialization)
    transition_points = Bool(False).tag(sync=True)
    transition_points_duration = Int(3000).tag(sync=True)
    prevent_filter_reset = Bool(False).tag(sync=True)
    selection = Array(default_value=None, allow_none=True).tag(
        sync=True, **ndarray_serialization
    )
    filter = Array(default_value=None, allow_none=True).tag(
        sync=True, **ndarray_serialization
    )
    hovering = Int(None, allow_none=True).tag(sync=True)

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

    # Interaction properties
    mouse_mode = Enum(['panZoom', 'lasso', 'rotate'], default_value='panZoom').tag(
        sync=True
    )
    lasso_type = Enum(['freeform', 'brush', 'rectangle'], default_value='freeform').tag(
        sync=True
    )
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
    tooltip_histograms = Bool().tag(sync=True)
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

    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = data
        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, event: dict, buffers):
        if event['type'] == EVENT_TYPES['TOOLTIP'] and isinstance(
            self.data, pd.DataFrame
        ):
            data = self.data.iloc[event['index']]
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
        data = self.data.iloc[point_idx]
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

    def create_download_view_button(self, icon_only=True, width=36):
        button = Button(
            description='' if icon_only else 'Download View',
            tooltip='Download View as PNG',
            icon='download',
            width=width,
        )

        def click_handler(event):
            self.send(
                {
                    'type': EVENT_TYPES['VIEW_DOWNLOAD'],
                    'transparentBackgroundColor': bool(event['alt_key']),
                }
            )

        button.on_click(click_handler)
        return button

    def create_save_view_button(self, icon_only=True, width=36):
        button = Button(
            description='' if icon_only else 'Save View',
            tooltip='Save View to Widget Property',
            icon='camera',
            width=width,
        )

        def click_handler(event):
            self.send(
                {
                    'type': EVENT_TYPES['VIEW_SAVE'],
                    'transparentBackgroundColor': bool(event['alt_key']),
                }
            )

        button.on_click(click_handler)
        return button

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

    def create_reset_view_button(self, icon_only=True, width=36):
        button = Button(
            description='' if icon_only else 'Reset View',
            icon='refresh',
            tooltip='Reset View',
            width=width,
        )

        def click_handler(event):
            self.reset_view(500, event['alt_key'])

        button.on_click(click_handler)
        return button

    def create_full_screen_button(self, icon_only=True, width=36):
        button = Button(
            description='' if icon_only else 'Full Screen',
            icon='expand',
            tooltip='Full Screen',
            width=width,
        )

        def click_handler(event):
            self.send({'type': EVENT_TYPES['FULL_SCREEN']})

        button.on_click(click_handler)
        return button

    def create_mouse_mode_toggle_button(self, mouse_mode, icon, tooltip, width=36):
        button = Button(
            description='',
            icon=icon,
            tooltip=tooltip,
            width=width,
            style='primary' if self.mouse_mode == mouse_mode else '',
        )

        def click_handler(b):
            button.style = 'primary'
            self.mouse_mode = mouse_mode

        def change_handler(change):
            button.style = 'primary' if change['new'] == mouse_mode else ''

        self.observe(change_handler, names=['mouse_mode'])

        button.on_click(click_handler)
        return button

    def create_lasso_type_button(self):
        button = ButtonChoice(
            icon={
                'freeform': '<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg"><path stroke-width="2px" stroke="currentColor" fill="none" d="m15.99958,27.5687c-1.8418,-0.3359 -3.71385,-1.01143 -5.49959,-2.04243c-6.69178,-3.8635 -9.65985,-11.26864 -6.62435,-16.52628c3.0355,-5.25764 10.93258,-6.38978 17.62435,-2.52628c6.1635,3.5585 9.16819,10.12222 7.23126,15.24508"/><circle stroke-width="2px" stroke="currentColor" fill="none" r="3" cy="25" cx="27"/></svg>',
                'brush': '<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg"><path stroke-width="2px" stroke="currentColor" fill="none" d="m25.985,26.755c-5.345,2.455 -10.786,2.981 -14.455,1.899c-3.449,-1.017 -5.53,-3.338 -5.53,-6.654c0,-3.33 1.705,-4.929 3.835,-6.127c0.894,-0.503 1.88,-0.912 2.738,-1.451c0.786,-0.493 1.427,-1.143 1.427,-2.422c0,-1.692 -1.552,-2.769 -3.177,-3.649c-3.177,-1.722 -7.152,-2.378 -7.152,-2.378l0.658,-3.946c0,0 4.665,0.784 8.4,2.806c2.987,1.618 5.271,4.055 5.271,7.167c0,3.33 -1.705,4.929 -3.835,6.127c-0.894,0.503 -1.88,0.912 -2.738,1.451c-0.786,0.493 -1.427,1.143 -1.427,2.422c0,1.486 1.117,2.362 2.662,2.818c2.897,0.854 7.122,0.332 11.338,-1.551"/><circle stroke-width="2px" stroke="currentColor" fill="none" r="3" cy="24" cx="27"/></svg>',
                'rectangle': '<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg"><circle  stroke-width="2px" stroke="currentColor" fill="none" r="3" cy="24" cx="27"/><path stroke-linecap="square" stroke-width="2px" stroke="currentColor" fill="none" d="m24,24l-22,0l0,-19l25,0l0,16"/></svg>',
            },
            tooltip='Lasso Type',
            width=36,
            value=self.lasso_type,
            options={
                'freeform': 'Freeform',
                'brush': 'Brush',
                'rectangle': 'Rectangle',
            },
        )

        def internal_change_handler(change):
            self.lasso_type = change['new']

        button.observe(internal_change_handler, names=['value'])

        def change_handler(change):
            button.value = change['new']

        self.observe(change_handler, names=['lasso_type'])

        return button

    def create_lasso_brush_size_button(self):
        button = ButtonIntSlider(
            icon='<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg"><circle fill="currentColor" r="3" cx="5" cy="16" /><circle fill="currentColor" r="9" cx="21" cy="16" /><path stroke-dasharray="0.5,1.5,0,0,0,0" stroke-width="1px" stroke="currentColor" fill="none" d="m5,13.717l12.378,-5.38l0.031,15.353l-12.409,-5.363l0,-4.61z"/></svg>',
            tooltip='Brush Size',
            width=36,
            slider_label='Brush Size',
            slider_label_value_suffix='px',
            slider_label_width=32,
            slider_width=128,
            value=self.lasso_brush_size,
            value_min=BRUSH_SIZE_MIN,
            value_max=BRUSH_SIZE_MAX,
            value_step=1,
        )

        def internal_change_handler(change):
            self.lasso_brush_size = change['new']

        button.observe(internal_change_handler, names=['value'])

        def change_handler(change):
            button.value = change['new']

        self.observe(change_handler, names=['lasso_brush_size'])

        return button

    def show(
        self, buttons: t.Optional[t.Union[t.List[WidgetButtons], Undefined]] = UNDEF
    ):
        button_pan_zoom = self.create_mouse_mode_toggle_button(
            mouse_mode='panZoom',
            icon='arrows',
            tooltip='Activate pan & zoom',
        )
        button_pan_zoom.disabled = self.camera_is_fixed
        button_lasso = self.create_mouse_mode_toggle_button(
            mouse_mode='lasso',
            icon='crosshairs',
            tooltip='Activate lasso selection',
        )
        # Hide the rotate button for now until we find a robust way to only use
        # it while axes are hidden.
        # button_rotate = self.create_mouse_mode_toggle_button(
        #     mouse_mode='rotate',
        #     icon='undo',
        #     tooltip='Activate rotation',
        # )
        button_view_save = self.create_save_view_button()
        button_view_download = self.create_download_view_button()
        button_view_reset = self.create_reset_view_button()
        button_full_screen = self.create_full_screen_button()
        button_lasso_type = self.create_lasso_type_button()
        button_lasso_brush_size = self.create_lasso_brush_size_button()
        button_lasso_brush_size.visible = self.lasso_type == 'brush'

        def lasso_type_change_handler(change):
            button_lasso_brush_size.visible = change['new'] == 'brush'

        self.observe(lasso_type_change_handler, names=['lasso_type'])

        button_map = {
            'pan_zoom': button_pan_zoom,
            'lasso': button_lasso,
            'save': button_view_save,
            'download': button_view_download,
            'reset': button_view_reset,
            'full_screen': button_full_screen,
            'divider': divider,
            'lasso_type': button_lasso_type,
            'lasso_brush_size': button_lasso_brush_size,
        }

        if buttons is not UNDEF:
            button_widgets = [
                button_map[button] for button in buttons if button in button_map
            ]
        else:
            button_widgets = [
                button_pan_zoom,
                button_lasso,
                button_lasso_type,
                button_lasso_brush_size,
                # button_rotate,
                divider,
                button_full_screen,
                # button_view_save,
                button_view_download,
                button_view_reset,
            ]

        buttons = widgets.VBox(
            children=button_widgets,
            layout=widgets.Layout(
                display='flex', flex_flow='column', align_items='stretch', width='40px'
            ),
        )

        plots = widgets.VBox(
            children=[self], layout=widgets.Layout(flex='1', width='auto')
        )

        def camera_is_fixed_change_handler(change):
            button_pan_zoom.disabled = change['new']

        self.observe(camera_is_fixed_change_handler, names=['camera_is_fixed'])

        return widgets.VBox([widgets.HBox([buttons, plots])])
