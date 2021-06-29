import base64
import IPython.display as ipydisplay
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from traitlets import Bool, Dict, Enum, Float, Int, List, Unicode, Union
from traittypes import Array

from ._version import __version__
from .utils import to_hex, with_left_label

SELECTION_DTYPE = 'uint32'

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
    return {'buffer': memoryview(ar), 'dtype': str(ar.dtype), 'shape': ar.shape}

def binary_to_array(value, obj=None):
    return np.frombuffer(value['data'], dtype=value['dtype']).reshape(value['shape'])

ndarray_serialization = dict(to_json=array_to_binary, from_json=binary_to_array)

@widgets.register
class JupyterScatter(widgets.DOMWidget):
    _view_name = Unicode("JupyterScatterView").tag(sync=True)
    _model_name = Unicode("JupyterScatterModel").tag(sync=True)
    _view_module = Unicode("jupyter-scatter").tag(sync=True)
    _model_module = Unicode("jupyter-scatter").tag(sync=True)
    _view_module_version = Unicode(__version__).tag(sync=True)
    _model_module_version = Unicode(__version__).tag(sync=True)
    _model_data = List([]).tag(sync=True)

    # For debugging
    dom_element_id = Unicode(read_only=True).tag(sync=True)

    # Data
    points = Array(default_value=None).tag(sync=True, **ndarray_serialization)
    x_domain = List(minlen=2, maxlen=2).tag(sync=True)
    y_domain = List(minlen=2, maxlen=2).tag(sync=True)
    selection = Array(default_value=None, allow_none=True).tag(sync=True, **ndarray_serialization)
    hovering = Int(None, allow_none=True).tag(sync=True)

    # View properties
    camera_target = List().tag(sync=True)
    camera_distance = Float().tag(sync=True)
    camera_rotation = Float().tag(sync=True)
    camera_view = List(None, allow_none=True).tag(sync=True)

    # Interaction properties
    mouse_mode = Enum(['panZoom', 'lasso', 'rotate'], default_value='panZoom').tag(sync=True)
    lasso_initiator = Bool().tag(sync=True)

    # Options
    color = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    color_active = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    color_hover = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    color_by = Enum([None, 'valueA', 'valueB'], allow_none=True, default_value=None).tag(sync=True)
    opacity = Union([Float(), List(Float())], allow_none=True).tag(sync=True)
    opacity_by = Enum([None, 'valueA', 'valueB', 'density'], allow_none=True, default_value=None).tag(sync=True)
    size = Union([Union([Int(), Float()]), List(Union([Int(), Float()]))]).tag(sync=True)
    size_active = Int().tag(sync=True)
    size_by = Enum([None, 'valueA', 'valueB'], allow_none=True, default_value=None).tag(sync=True)
    connect = Bool().tag(sync=True)
    connection_color = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    connection_color_active = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    connection_color_hover = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    connection_color_by = Enum([None, 'valueA', 'valueB', 'segment'], allow_none=True, default_value=None).tag(sync=True)
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

    # For any kind of options. Note that whatever is defined in options will
    # be overwritten by the short-hand options
    other_options = Dict(dict()).tag(sync=True)

    view_reset = Bool(False).tag(sync=True) # Used for triggering a view reset
    view_download = Unicode(None, allow_none=True).tag(sync=True) # Used for triggering a download
    view_pixels = List(None, allow_none=True, read_only=True).tag(sync=True)
    view_shape = List(None, allow_none=True, read_only=True).tag(sync=True)

    @property
    def mouse_mode_widget(self):
        widget = widgets.RadioButtons(
            options=[
                ('Pan & zoom', 'panZoom'),
                ('Lasso selection', 'lasso'),
                ('Rotation', 'rotate')
            ],
            value='panZoom'
        )

        def change_handler(change):
            self.mouse_mode = change.new

        widget.observe(change_handler, names='value')

        return with_left_label('Mouse mode', widget)

    @property
    def lasso_initiator_widget(self):
        widget = widgets.Checkbox(
            icon='check',
            indent=False,
            value=self.lasso_initiator
        )
        widgets.jslink((self, 'lasso_initiator'), (widget, 'value'))
        return with_left_label('Enable lasso initiator', widget)

    @property
    def selection_widget(self):
        widget = widgets.Textarea(
            value=', '.join(f'{n}' for n in self.selection),
            placeholder='The indices of selected points will be shown here',
            disabled=True
        )

        def change_handler(change):
            widget.value = ', '.join(f'{n}' for n in change.new)

        self.observe(change_handler, names='selected_points')

        return with_left_label('Selected points', widget)

    @property
    def hovering_widget(self):
        widget = widgets.Text(
            value=f'{self.hovered_point}',
            placeholder='The index of the hovered point will be shown here',
            disabled=True
        )

        def change_handler(change):
            widget.value = f'{change.new}'

        self.observe(change_handler, names='hovered_point')

        return with_left_label('Hovered point', widget)

    @property
    def color_widgets(self):
        color_by_widget = widgets.RadioButtons(
            options=[
                ('Point color', 'none'),
                ('Category (using colormap)', 'category'),
                ('Value (using colormap)', 'value')
            ],
            value='none' if self.color_by is None else self.color_by
        )

        colormap_widget = widgets.Combobox(
            placeholder='Choose a Matplotlib colormap',
            options=list(plt.colormaps()),
            ensure_option=True,
            disabled=self.color_by is None,
        )

        color_widget = widgets.ColorPicker(
            value=to_hex(self.color),
            disabled=False
        )

        def color_by_change_handler(change):
            if change.new == 'none':
                self.color_by = None
                self.color = color_widget.value

                colormap_widget.disabled = True
                color_widget.disabled = False

            else:
                if change.new == 'category':
                    colormap_widget.options = [
                        'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1',
                        'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
                    ]
                else:
                    colormap_widget.options = [
                        'greys', 'viridis', 'plasma', 'inferno', 'magma',
                        'cividis', 'coolwarm', 'RdGy'
                    ]

                self.color_by = change.new
                if colormap_widget.value:
                    self.use_cmap(colormap_widget.value)

                colormap_widget.disabled = False
                color_widget.disabled = True

        def colormap_change_handler(change):
            self.use_cmap(change.new)

        def color_change_handler(change):
            self.color = change.new

        color_by_widget.observe(color_by_change_handler, names='value')
        colormap_widget.observe(colormap_change_handler, names='value')
        color_widget.observe(color_change_handler, names='value')

        return (
            with_left_label('Color by', color_by_widget),
            with_left_label('Colormap', colormap_widget),
            with_left_label('Point color', color_widget)
        )

    @property
    def color_by_widget(self):
        widget = widgets.RadioButtons(
            options=[
                ('Point color', 'none'),
                ('Category (using colormap)', 'category'),
                ('Value (using colormap)', 'value')
            ],
            value='none'
        )

        def change_handler(change):
            if change.new == 'none':
                self.color_by = None
            else:
                self.color_by = change.new

        widget.observe(change_handler, names='value')

        return with_left_label('Color by', widget)

    @property
    def color_map_widget(self):
        widget = widgets.Combobox(
            placeholder='Choose a Matplotlib colormap',
            options=list(plt.colormaps()),
            ensure_option=True,
            disabled=True,
        )

        def change_handler(change):
            self.use_cmap(change.new)

        widget.observe(change_handler, names='value')

        return with_left_label('Color map', widget)

    @property
    def height_widget(self):
        widget = widgets.IntSlider(
            value=self.height,
            min=128,
            max=max(1280, self.height + 64),
            step=32
        )
        widgets.jslink((self, 'height'), (widget, 'value'))
        return with_left_label('Height', widget)

    @property
    def background_color_widget(self):
        widget = widgets.ColorPicker(value=to_hex(self.background_color))

        def change_handler(change):
            self.background_color = change.new

        widget.observe(change_handler, names='value')

        return with_left_label('Background color', widget)

    @property
    def background_image_widget(self):
        widget = widgets.FileUpload(
            accept='image/*',
            description='Select a file',
            multiple=False
        )

        def change_handler(change):
            [filename] = change.new
            ext = filename.split('.')[-1]
            prefix = f'data:image/{ext};base64,'
            filedata = change.new[filename]['content']
            self.background_image = f'{prefix}{base64.b64encode(filedata).decode("utf-8")}'

        widget.observe(change_handler, names='value')

        return with_left_label('Background image', widget)

    @property
    def lasso_color_widget(self):
        widget = widgets.ColorPicker(value=to_hex(self.lasso_color))

        def change_handler(change):
            self.lasso_color = change.new

        widget.observe(change_handler, names='value')

        return with_left_label('Lasso color', widget)

    @property
    def lasso_min_delay_widget(self):
        widget = widgets.IntSlider(
            value=self.lasso_min_delay,
            min=0,
            max=max(100, self.lasso_min_delay),
            step=5
        )
        widgets.jslink((self, 'lasso_min_delay'), (widget, 'value'))
        return with_left_label('Lasso min delay', widget)

    @property
    def lasso_min_dist_widget(self):
        widget = widgets.IntSlider(
            value=self.lasso_min_dist,
            min=0,
            max=max(32, self.lasso_min_dist),
            step=2
        )
        widgets.jslink((self, 'lasso_min_dist'), (widget, 'value'))
        return with_left_label('Lasso min dist', widget)

    @property
    def color_widget(self):
        widget = widgets.ColorPicker(value=to_hex(self.color))

        def change_handler(change):
            self.color = change.new

        widget.observe(change_handler, names='value')

        return with_left_label('Point color', widget)

    @property
    def color_active_widget(self):
        widget = widgets.ColorPicker(value=to_hex(self.color_active))

        def change_handler(change):
            self.color_active = change.new

        widget.observe(change_handler, names='value')

        return with_left_label('Point color active', widget)

    @property
    def color_hover_widget(self):
        widget = widgets.ColorPicker(value=to_hex(self.color_hover))

        def change_handler(change):
            self.color_hover = change.new

        widget.observe(change_handler, names='value')

        return with_left_label('Point color hover', widget)

    @property
    def opacity_widget(self):
        widget = widgets.FloatSlider(
            value=self.opacity,
            min=0.0,
            max=1.0,
            step=0.05,
            continuous_update=True
        )
        widgets.jslink((self, 'opacity'), (widget, 'value'))
        return with_left_label('Point opacity', widget)

    @property
    def selection_outline_width_widget(self):
        widget = widgets.IntSlider(
            value=self.selection_outline_width,
            min=0,
            max=max(8, self.selection_outline_width + 2),
            continuous_update=True
        )
        widgets.jslink((self, 'selection_outline_width'), (widget, 'value'))
        return with_left_label('Point outline width', widget)

    @property
    def size_widget(self):
        widget = widgets.IntSlider(
            value=self.size,
            min=1,
            max=max(10, self.size + 5),
            continuous_update=True
        )
        widgets.jslink((self, 'size'), (widget, 'value'))
        return with_left_label('Point size', widget)

    @property
    def selection_size_addition_widget(self):
        widget = widgets.IntSlider(
            value=self.selection_size_addition,
            min=0,
            max=max(8, self.selection_size_addition + 2),
            continuous_update=True
        )
        widgets.jslink((self, 'selection_size_addition'), (widget, 'value'))
        return with_left_label('Point size selected', widget)

    @property
    def reticle_widget(self):
        widget = widgets.Checkbox(
            icon='check',
            indent=False,
            value=self.reticle
        )
        widgets.jslink((self, 'reticle'), (widget, 'value'))
        return with_left_label('Show reticle', widget)

    @property
    def reticle_color_widget(self):
        widget = widgets.ColorPicker(value=to_hex(self.reticle_color))

        def change_handler(change):
            self.reticle_color = change.new

        widget.observe(change_handler, names='value')

        return with_left_label('Reticle color', widget)

    def get_download_view_widget(self, icon_only=False, width=None):
        button = widgets.Button(
            description='' if icon_only else 'Download View',
            icon='download'
        )

        if width is not None:
            button.layout.width = f'{width}px'

        def click_handler(b):
            self.download_view('file')

        button.on_click(click_handler)
        return button

    @property
    def download_view_widget(self):
        return self.get_download_view_widget()

    def get_save_view_widget(self, icon_only=False, width=None):
        button = widgets.Button(
            description='' if icon_only else 'Save View',
            icon='camera'
        )

        if width is not None:
            button.layout.width = f'{width}px'

        def click_handler(b):
            self.download_view('property')

        button.on_click(click_handler)
        return button

    @property
    def save_view_widget(self):
        return self.get_save_view_widget()

    def get_reset_view_widget(self, icon_only=False, width=None):
        button = widgets.Button(
            description='' if icon_only else 'Reset View',
            icon='refresh'
        )

        if width is not None:
            button.layout.width = f'{width}px'

        def click_handler(b):
            self.reset_view()

        button.on_click(click_handler)
        return button

    @property
    def reset_view_widget(self):
        return self.get_reset_view_widget()

    def get_separator(self, color='#efefef', margin_top=10, margin_bottom=10):
        return widgets.Box(
            children=[],
            layout=widgets.Layout(
                margin=f'{margin_top}px 0 {margin_bottom}px 0',
                width='100%',
                height='0',
                border=f'1px solid {color}'
            )
        ),

    def options(self):
        """Display widgets for all options
        """
        color_by_widget, colormap_widget, color_widget = self.color_widgets
        ipydisplay.display(
            self.mouse_mode_widget,
            self.lasso_initiator_widget,
            self.height_widget,
            self.size_widget,
            self.opacity_widget,
            color_by_widget,
            colormap_widget,
            color_widget,
            self.color_active_widget,
            self.color_hover_widget,
            self.selection_outline_width_widget,
            self.selection_size_addition_widget,
            self.lasso_color_widget,
            self.reticle_color_widget,
            self.reticle_widget,
            self.background_color_widget,
            self.background_image_widget,
            widgets.Box(
                children=[],
                layout=widgets.Layout(
                    margin='10px 0',
                    width='100%',
                    height='0',
                    border='1px solid #efefef'
                )
            ),
            self.reset_view_widget
        )

    def use_cmap(self, cmap_name: str, reverse: bool = False):
        """Use a Matplotlib colormap for the point color.

        Parameters
        ----------
        cmap_name : str
            The name of the Matplotlib color map.
        reverse : bool, optional
            Reverse the colormap when set to ``True``.
        """
        self.color = plt.get_cmap(cmap_name)(range(256)).tolist()[::(1 + (-2 * reverse))]

    def reset_view(self):
        self.view_reset = True

    def download_view(self, target = 'file'):
        self.view_download = target

    def select(self, points):
        """Select points

        Parameters
        ----------
        points : 1D array_like
            List of point indices to select
        """

        self.selection = np.asarray(points).astype(SELECTION_DTYPE)

    def get_panzoom_mode_widget(self, icon_only=False, width=None):
        button = widgets.Button(
            description='' if icon_only else 'Pan & Zoom',
            icon='arrows',
            tooltip='Activate pan & zoom',
            button_style = 'primary' if self.mouse_mode == 'panZoom' else '',
        )

        if width is not None:
            button.layout.width = f'{width}px'

        def click_handler(b):
            button.button_style = 'primary'
            self.mouse_mode = 'panZoom'

        def change_handler(change):
            button.button_style = 'primary' if change['new'] == 'panZoom' else ''

        self.observe(change_handler, names=['mouse_mode'])

        button.on_click(click_handler)
        return button

    def get_lasso_mode_widget(self, icon_only=False, width=None):
        button = widgets.Button(
            description='' if icon_only else 'Lasso',
            icon='crosshairs',
            tooltip='Activate lasso selection',
            button_style = 'primary' if self.mouse_mode == 'lasso' else '',
        )

        if width is not None:
            button.layout.width = f'{width}px'

        def click_handler(b):
            button.button_style = 'primary'
            self.mouse_mode = 'lasso'

        def change_handler(change):
            button.button_style = 'primary' if change['new'] == 'lasso' else ''

        self.observe(change_handler, names=['mouse_mode'])

        button.on_click(click_handler)
        return button

    def get_rotate_mode_widget(self, icon_only=False, width=None):
        button = widgets.Button(
            description='' if icon_only else 'Rotate',
            icon='undo',
            tooltip='Activate rotation',
            button_style = 'primary' if self.mouse_mode == 'rotate' else '',
        )

        if width is not None:
            button.layout.width = f'{width}px'

        def click_handler(b):
            button.button_style = 'primary'
            self.mouse_mode = 'rotate'

        def change_handler(change):
            button.button_style = 'primary' if change['new'] == 'rotate' else ''

        self.observe(change_handler, names=['mouse_mode'])

        button.on_click(click_handler)
        return button

    def show(self):
        buttons = widgets.VBox(
            children=[
                self.get_panzoom_mode_widget(icon_only=True, width=36),
                self.get_lasso_mode_widget(icon_only=True, width=36),
                self.get_rotate_mode_widget(icon_only=True, width=36),
               widgets.Box(
                    children=[],
                    layout=widgets.Layout(
                        margin='10px 0',
                        width='100%',
                        height='0',
                        border='1px solid #efefef'
                    )
                ),
                self.get_save_view_widget(icon_only=True, width=36),
                self.get_download_view_widget(icon_only=True, width=36),
                self.get_reset_view_widget(icon_only=True, width=36),
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
