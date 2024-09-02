import base64
import IPython.display as ipydisplay
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import anywidget
import pandas as pd
import pathlib

from traitlets import Bool, Dict, Enum, Float, Int, List, Unicode, Union
from traittypes import Array

from .annotations_traits import (
    Line,
    HLine,
    VLine,
    Rect,
    serialization as annotation_serialization,
)

SELECTION_DTYPE = 'uint32'
EVENT_TYPES = {
    "TOOLTIP": "tooltip",
    "VIEW_DOWNLOAD": "view_download",
    "VIEW_RESET": "view_reset",
    "VIEW_SAVE": "view_save",
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

# Code extracted from maartenbreddels ipyvolume
def array_to_binary(ar, obj=None, force_contiguous=True):
    if ar is None:
        return None
    if ar.dtype.kind not in ['u', 'i', 'f']:  # ints and floats
        raise ValueError("unsupported dtype: %s" % (ar.dtype))
    if ar.dtype == np.float64:  # WebGL does not support float64, case it here
        ar = ar.astype(np.float32)
    if ar.dtype == np.int64:  # JS does not support int64
        ar = ar.astype(np.int32)
    if force_contiguous and not ar.flags["C_CONTIGUOUS"]:  # make sure it's contiguous
        ar = np.ascontiguousarray(ar)
    return {'view': memoryview(ar), 'dtype': str(ar.dtype), 'shape': ar.shape}

def binary_to_array(value, obj=None):
    return np.frombuffer(value['view'], dtype=value['dtype']).reshape(value['shape'])

ndarray_serialization = dict(to_json=array_to_binary, from_json=binary_to_array)

class JupyterScatter(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "bundle.js"

    # For debugging
    dom_element_id = Unicode(read_only=True).tag(sync=True)

    # Data
    points = Array(default_value=None).tag(sync=True, **ndarray_serialization)
    transition_points = Bool(False).tag(sync=True)
    transition_points_duration = Int(3000).tag(sync=True)
    prevent_filter_reset = Bool(False).tag(sync=True)
    selection = Array(default_value=None, allow_none=True).tag(sync=True, **ndarray_serialization)
    filter = Array(default_value=None, allow_none=True).tag(sync=True, **ndarray_serialization)
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
    color_domain = Union([Dict(), List(minlen=2, maxlen=2)], allow_none=True).tag(sync=True)
    opacity_domain = Union([Dict(), List(minlen=2, maxlen=2)], allow_none=True).tag(sync=True)
    size_domain = Union([Dict(), List(minlen=2, maxlen=2)], allow_none=True).tag(sync=True)

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
    color_histogram_range = List(None, allow_none=True, minlen=2, maxlen=2).tag(sync=True)
    opacity_histogram_range = List(None, allow_none=True, minlen=2, maxlen=2).tag(sync=True)
    size_histogram_range = List(None, allow_none=True, minlen=2, maxlen=2).tag(sync=True)

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

    # Zoom properties
    zoom_to = Array(default_value=None, allow_none=True).tag(sync=True, **ndarray_serialization)
    zoom_to_call_idx = Int(0).tag(sync=True)
    zoom_animation = Int(1000).tag(sync=True)
    zoom_padding = Float(0.333).tag(sync=True)
    zoom_on_selection = Bool(False).tag(sync=True)
    zoom_on_filter = Bool(False).tag(sync=True)

    # Interaction properties
    mouse_mode = Enum(['panZoom', 'lasso', 'rotate'], default_value='panZoom').tag(sync=True)
    lasso_initiator = Bool().tag(sync=True)
    lasso_on_long_press = Bool().tag(sync=True)

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
        default_value='top-left'
    ).tag(sync=True)
    legend_size = Enum(
        ['small', 'medium', 'large'], default_value='small'
    ).tag(sync=True)
    legend_color = List(
        default_value=[0, 0, 0, 1], minlen=4, maxlen=4
    ).tag(sync=True)
    legend_encoding = Dict(dict()).tag(sync=True)

    # Tooltip
    tooltip_enable = Bool().tag(sync=True)
    tooltip_size = Enum(
        ['small', 'medium', 'large'], default_value='small'
    ).tag(sync=True)
    tooltip_color = List(
        default_value=[0, 0, 0, 1], minlen=4, maxlen=4
    ).tag(sync=True)
    tooltip_properties = List(
        default_value=['x', 'y', 'color', 'opacity', 'size']
    ).tag(sync=True)
    tooltip_properties_non_visual_info = Dict(dict()).tag(sync=True)
    tooltip_histograms = Bool().tag(sync=True)
    tooltip_histograms_ranges = Dict(dict()).tag(sync=True)
    tooltip_histograms_size = Enum(
        ['small', 'medium', 'large'], default_value='small'
    ).tag(sync=True)
    tooltip_preview = Unicode(None, allow_none=True).tag(sync=True)
    tooltip_preview_type = Enum(
        ['text', 'image', 'audio'], default_value='text'
    ).tag(sync=True)
    tooltip_preview_text_lines = Int(default_value=3, allow_none=True).tag(sync=True)
    tooltip_preview_image_background_color = Union([Enum(['auto']), Unicode()], default_value='auto').tag(sync=True)
    tooltip_preview_image_position = Union([Enum(['top', 'left', 'right', 'bottom', 'center']), Unicode()], allow_none=True, default_value=None).tag(sync=True)
    tooltip_preview_image_size = Enum(['contain', 'cover'], allow_none=True, default_value=None).tag(sync=True)
    tooltip_preview_audio_length = Int(None, allow_none=True).tag(sync=True)
    tooltip_preview_audio_loop = Bool().tag(sync=True)
    tooltip_preview_audio_controls = Bool().tag(sync=True)

    # Options
    color = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    color_selected = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    color_hover = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    color_by = Enum([None, 'valueA', 'valueB'], allow_none=True, default_value=None).tag(sync=True)
    opacity = Union([Float(), List(Float())], allow_none=True).tag(sync=True)
    opacity_unselected = Float().tag(sync=True)
    opacity_by = Enum([None, 'valueA', 'valueB', 'density'], allow_none=True, default_value=None).tag(sync=True)
    size = Union([Union([Int(), Float()]), List(Union([Int(), Float()]))]).tag(sync=True)
    size_by = Enum([None, 'valueA', 'valueB'], allow_none=True, default_value=None).tag(sync=True)
    connect = Bool().tag(sync=True)
    connection_color = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    connection_color_selected = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    connection_color_hover = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    connection_color_by = Enum([None, 'valueA', 'valueB', 'segment', 'inherit'], allow_none=True, default_value=None).tag(sync=True)
    connection_opacity = Union([Float(), List(Float())], allow_none=True).tag(sync=True)
    connection_opacity_by = Enum([None, 'valueA', 'valueB', 'segment'], allow_none=True, default_value=None).tag(sync=True)
    connection_size = Union([Union([Int(), Float()]), List(Union([Int(), Float()]))]).tag(sync=True)
    connection_size_by = Enum([None, 'valueA', 'valueB', 'segment'], allow_none=True, default_value=None).tag(sync=True)
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

    view_download = Unicode(None, allow_none=True).tag(sync=True) # Used for triggering a download
    view_data = Array(default_value=None, allow_none=True, read_only=True).tag(sync=True, **ndarray_serialization)

    # For synchronyzing view changes across scatter plot instances
    view_sync = Unicode(None, allow_none=True).tag(sync=True)

    event_types = Dict(EVENT_TYPES, read_only=True).tag(sync=True)

    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = data
        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, event: dict, buffers):
        if event["type"] == EVENT_TYPES["TOOLTIP"] and isinstance(self.data, pd.DataFrame):
            data = self.data.iloc[event["index"]]
            self.send({
                "type": EVENT_TYPES["TOOLTIP"],
                "index": event["index"],
                "preview": data[event["preview"]] if event["preview"] is not None else None,
                "properties": data[event["properties"]].to_dict()
            })

    def show_tooltip(self, point_idx):
        data = self.data.iloc[point_idx]
        self.send({
            "type": TOOLTIP_EVENT_TYPE,
            "show": True,
            "index": point_idx,
            "preview": data[self.tooltip_preview] if self.tooltip_preview is not None else None,
            "properties": data[self.tooltip_properties_non_visual_info.keys()].to_dict() if self.tooltip_properties_non_visual_info is not None else {}
        })

    def create_download_view_button(self, icon_only=True, width=36):
        button = Button(
            description='' if icon_only else 'Download View',
            tooltip='Download View as PNG',
            icon='download',
            width=width,
        )

        def click_handler(event):
            self.send({
                "type": EVENT_TYPES["VIEW_DOWNLOAD"],
                "transparentBackgroundColor": bool(event["alt_key"]),
            })

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
            self.send({
                "type": EVENT_TYPES["VIEW_SAVE"],
                "transparentBackgroundColor": bool(event["alt_key"]),
            })

        button.on_click(click_handler)
        return button

    def reset_view(self, animation: int = 0, data_extent: bool = False):
        if data_extent:
            self.send({
                "type": EVENT_TYPES["VIEW_RESET"],
                "area": {
                    "x": self.points[:, 0].min(),
                    "width": self.points[:, 0].max() - self.points[:, 0].min(),
                    "y": self.points[:, 1].min(),
                    "height": self.points[:, 1].max() - self.points[:, 1].min(),
                },
                "animation": animation
            })
        else:
            self.send({
                "type": EVENT_TYPES["VIEW_RESET"],
                "animation": animation
            })

    def create_reset_view_button(self, icon_only=True, width=36):
        button = Button(
            description='' if icon_only else 'Reset View',
            icon='refresh',
            tooltip='Reset View',
            width=width,
        )

        def click_handler(event):
            self.reset_view(500, event["alt_key"])

        button.on_click(click_handler)
        return button

    def create_mouse_mode_toggle_button(
        self,
        mouse_mode,
        icon,
        tooltip,
    ):
        button = widgets.Button(
            description='',
            icon=icon,
            tooltip=tooltip,
            button_style = 'primary' if self.mouse_mode == mouse_mode else '',
        )

        button.layout.width = '36px'

        def click_handler(b):
            button.button_style = 'primary'
            self.mouse_mode = mouse_mode

        def change_handler(change):
            button.button_style = 'primary' if change['new'] == mouse_mode else ''

        self.observe(change_handler, names=['mouse_mode'])

        button.on_click(click_handler)
        return button

    def show(self):
        button_pan_zoom = self.create_mouse_mode_toggle_button(
            mouse_mode='panZoom',
            icon='arrows',
            tooltip='Activate pan & zoom',
        )
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

        buttons = widgets.VBox(
            children=[
                button_pan_zoom,
                button_lasso,
                # button_rotate,
                widgets.Box(
                    children=[],
                    layout=widgets.Layout(
                        margin='10px 0',
                        width='100%',
                        height='0',
                        border='1px solid var(--jp-layout-color2)'
                    )
                ),
                button_view_save,
                button_view_download,
                button_view_reset,
            ],
            layout=widgets.Layout(
                display='flex',
                flex_flow='column',
                align_items='stretch',
                width='40px'
            )
        )

        plots = widgets.VBox(
            children=[self],
            layout=widgets.Layout(
                flex='1',
                width='auto'
            )
        )

        return widgets.HBox([buttons, plots])

class Button(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
      const button = document.createElement('button');

      button.classList.add('jupyter-widgets');
      button.classList.add('jupyter-button');
      button.classList.add('widget-button');

      const update = () => {
        const description = model.get('description');
        const icon = model.get('icon');
        const tooltip = model.get('tooltip');
        const width = model.get('width');

        button.textContent = '';

        if (icon) {
          const i = document.createElement('i');
          i.classList.add('fa', `fa-${icon}`);

          if (!description) {
            i.classList.add('center');
          }

          button.appendChild(i);
        }

        if (description) {
          button.appendChild(document.createTextNode(description));
        }

        if (tooltip) {
          button.title = tooltip;
        }

        if (width) {
          button.style.width = `${width}px`;
        }
      }

      const createEventHandler = (eventType) => (event) => {
        model.send({
          type: eventType,
          alt_key: event.altKey,
          shift_key: event.shiftKey,
          meta_key: event.metaKey,
        });
      }

      const clickHandler = createEventHandler('click');
      const dblclickHandler = createEventHandler('dblclick');

      button.addEventListener('click', clickHandler);
      button.addEventListener('dblclick', dblclickHandler);

      model.on('change:description', update);
      model.on('change:icon', update);
      model.on('change:width', update);
      model.on('change:tooltip', update);

      update();

      el.appendChild(button);

      return () => {
        button.removeEventListener('click', clickHandler);
        button.removeEventListener('dblclick', dblclickHandler);
      };
    }
    export default { render }
    """

    description = Unicode().tag(sync=True)
    icon = Unicode().tag(sync=True)
    width = Int(allow_none=True).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._click_handler = None
        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, event: dict, buffers):
        if event["type"] == "click" and self._click_handler is not None:
            self._click_handler(event)
        if event["type"] == "dblclick" and self._dblclick_handler is not None:
            self._dblclick_handler(event)

    def on_click(self, callback):
        self._click_handler = callback

    def on_dblclick(self, callback):
        self._dblclick_handler = callback
