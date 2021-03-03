import base64
import math
import IPython.display as ipydisplay
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import typing as t

from matplotlib.colors import to_rgba
from traitlets import Bool, Dict, Enum, Float, Int, List, Unicode, Union

from ._version import __version__
from .utils import to_hex, with_left_label

from .color_maps import okabe_ito, glasbey_light, glasbey_dark

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
    points = List().tag(sync=True)
    selection = List().tag(sync=True)
    hovering = Int(None, allow_none=True, read_only=True).tag(sync=True)

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
    opacity_by = Enum([None, 'valueA', 'valueB'], allow_none=True, default_value=None).tag(sync=True)
    size = Union([Union([Int(), Float()]), List(Union([Int(), Float()]))]).tag(sync=True)
    size_by = Enum([None, 'valueA', 'valueB'], allow_none=True, default_value=None).tag(sync=True)
    connect = Bool().tag(sync=True)
    connection_color = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    connection_color_active = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    connection_color_hover = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    connection_color_by = Enum([None, 'valueA', 'valueB'], allow_none=True, default_value=None).tag(sync=True)
    connection_opacity = Union([Float(), List(Float())], allow_none=True).tag(sync=True)
    connection_opacity_by = Enum([None, 'valueA', 'valueB'], allow_none=True, default_value=None).tag(sync=True)
    connection_size = Union([Union([Int(), Float()]), List(Union([Int(), Float()]))]).tag(sync=True)
    connection_size_by = Enum([None, 'valueA', 'valueB'], allow_none=True, default_value=None).tag(sync=True)
    height = Int().tag(sync=True)
    background_color = Union([Unicode(), List(minlen=4, maxlen=4)]).tag(sync=True)
    background_image = Unicode(None, allow_none=True).tag(sync=True)
    lasso_color = Union([Unicode(), List(minlen=4, maxlen=4)]).tag(sync=True)
    lasso_min_delay = Int().tag(sync=True)
    lasso_min_dist = Float().tag(sync=True)
    selection_size_addition = Int().tag(sync=True)
    selection_outline_width = Int().tag(sync=True)
    recticle = Bool().tag(sync=True)
    recticle_color = Union([Unicode(), List(minlen=4, maxlen=4)]).tag(sync=True)

    # For any kind of options. Note that whatever is defined in options will
    # be overwritten by the short-hand options
    other_options = Dict(dict()).tag(sync=True)

    # Used for triggering a view reset
    view_reset = Bool(False).tag(sync=True)

    # Needed when the user specified `connect_order`
    sort_order = Dict(None, allow_none=True).tag(sync=True)

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
    def recticle_widget(self):
        widget = widgets.Checkbox(
            icon='check',
            indent=False,
            value=self.recticle
        )
        widgets.jslink((self, 'recticle'), (widget, 'value'))
        return with_left_label('Show recticle', widget)

    @property
    def recticle_color_widget(self):
        widget = widgets.ColorPicker(value=to_hex(self.recticle_color))

        def change_handler(change):
            self.recticle_color = change.new

        widget.observe(change_handler, names='value')

        return with_left_label('Recticle color', widget)

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
            self.recticle_color_widget,
            self.recticle_widget,
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
                self.get_reset_view_widget(icon_only=True, width=36)
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


def plot(
    x: t.Union[str, np.ndarray],
    y: t.Union[str, np.ndarray],
    # Data
    data: pd.DataFrame = None,
    selection: t.Union[str, list, np.ndarray] = [],
    # Visual encoding
    color: t.Union[str, tuple[float, float, float], tuple[float, float, float, float]] = (0, 0, 0, 0.66),
    color_active: t.Union[str, tuple[float, float, float], tuple[float, float, float, float]] = (0, 0.55, 1, 1),
    color_hover: t.Union[str, tuple[float, float, float], tuple[float, float, float, float], np.ndarray] = (0, 0, 0, 1),
    color_by: t.Union[str, np.ndarray] = None,
    color_norm: t.Union[tuple[float, float], matplotlib.colors.Normalize] = None,
    color_order: list = None,
    color_map: t.List[t.Union[tuple[float, float, float], tuple[float, float, float, float]]] = None,
    opacity: float = 0.66,
    opacity_by: t.Union[str, np.ndarray] = None,
    opacity_norm: t.Union[tuple[float, float], matplotlib.colors.Normalize] = None,
    opacity_order: list = None,
    opacity_map: t.List[tuple[float]] = None,
    size: int = 4,
    size_by: t.Union[str, np.ndarray] = None,
    size_norm: t.Union[tuple[float, float], matplotlib.colors.Normalize] = None,
    size_order: list = None,
    size_map: t.List[tuple[int]] = None,
    connect_by: t.Union[str, np.ndarray] = None,
    connect_order: t.Union[str, np.ndarray] = None,
    connection_color: t.Union[str, tuple[float, float, float], tuple[float, float, float, float]] = (0, 0, 0, 0.1),
    connection_color_active: t.Union[str, tuple[float, float, float], tuple[float, float, float, float]] = (0, 0.55, 1, 1),
    connection_color_hover: t.Union[str, tuple[float, float, float], tuple[float, float, float, float], np.ndarray] = (0, 0, 0, 0.66),
    connection_color_by: t.Union[str, np.ndarray] = None,
    connection_color_norm: t.Union[tuple[float, float], matplotlib.colors.Normalize] = None,
    connection_color_order: list = None,
    connection_color_map: t.List[t.Union[tuple[float, float, float], tuple[float, float, float, float]]] = None,
    connection_opacity: float = 0.1,
    connection_opacity_by: t.Union[str, np.ndarray] = None,
    connection_opacity_norm: t.Union[tuple[float, float], matplotlib.colors.Normalize] = None,
    connection_opacity_order: list = None,
    connection_opacity_map: t.List[tuple[float]] = None,
    connection_size: int = 2,
    connection_size_by: t.Union[str, np.ndarray] = None,
    connection_size_norm: t.Union[tuple[float, float], matplotlib.colors.Normalize] = None,
    connection_size_order: list = None,
    connection_size_map: t.List[tuple[int]] = None,
    # Other properties
    background_color: t.Union[str, tuple[float, float, float], tuple[float, float, float, float]] = (1, 1, 1, 1),
    background_image: str = None,
    camera_distance: float = 1.0,
    camera_rotation: float = 0.0,
    camera_target: tuple[float, float] = [0, 0],
    camera_view: list = None,
    height: int = 240,
    lasso_color: t.Union[str, tuple[float, float, float], tuple[float, float, float, float]] = (0, 0, 0, 1),
    lasso_initiator: bool = True,
    lasso_min_delay: int = 10,
    lasso_min_dist: float = 3.0,
    mouse_mode: str = 'panZoom',
    recticle_color: t.Union[str, tuple[float, float, float], tuple[float, float, float, float]] = (0, 0.55, 1, 0.33),
    selection_outline_width: int = 2,
    selection_size_addition: int = 2,
    recticle: bool = True,
    options: dict = {},
):
    """Display a scatter widget

    Parameters
    ----------
    x : str or np.ndarray
        Either the name of column in ``data`` or a numpy array
    data : pd.dataframe, optional
        Pandas DataFrame
    values : 1D array_like, optional
        Numerical values associated to the ``points``
    color_by : str, optional
        Coloring option, which can either be ``None``, category, or value
    selectiom : 1D array_like, optional
        List of indices of points to be selected
    height : int, optional
        Height of the scatter plot
    background_color : list or quadruple or Matplotlib color, optional
        Background color of the scatter plot
    background_image : str, optional
        URL to a background image
    lasso_color : list or quadruple or Matplotlib color, optional
        Lasso color
    lasso_min_delay : int, optional
        Minimum delay in milliseconds before re-evaluating the lasso
    lasso_min_dist : float, optional
        Minimum distance in pixels before adding a new point to the lasso
    color : list or quadruple or Matplotlib color, optional
        Point color
    color_active : list, optional
        Description
    color_hover : list, optional
        Description
    opacity : float, optional
        Description
    size : int, optional
        Description
    selection_size_addition : int, optional
        Increase of the point size in pixel of selected points
    selection_outline_width : int, optional
        Width of the outline of selected points.
    recticle : bool, optional
        If ``True`` show the recticle upon hovering the mouse above a point
    recticle_color : list or quadruple or Matplotlib color, optional
        Recticle color
    camera_target : list, optional
        Camera center point in normalized device coordinates
    camera_distance : float, optional
        Camera distance
    camera_rotation : float, optional
        Camera rotation in degrees
    camera_view : list, optional
        Camera view as a 16-dimensional
    mouse_mode : str, optional
        Description
    lasso_initiator : bool, optional
        If ``True`` a lasso indicator will be shown upon clicking somewhere
        into the background
    options : dict, optional
        Key-value pairs of additional options for regl-scatterplot

    Returns
    -------
    scatterplot widget
    """
    n = None
    sort_order = None

    point_data_encodings = [color_by, opacity_by, size_by, connection_color_by, connection_opacity_by, connection_size_by]
    point_data_encodings = [x for x in point_data_encodings if x is not None]
    point_data_encodings = dict.fromkeys(point_data_encodings)

    assert len(list(point_data_encodings)) < 3, 'Only two point data encodings are currently supported'

    try:
        n = len(data)
    except TypeError:
        n = x.size

    ndim = 5 if connect_by is not None else 4
    points = np.zeros((n, ndim))

    try:
        points[:, 0] = data[x].values
    except TypeError:
        points[:, 0] = np.asarray(x)

    try:
        points[:, 1] = data[y].values
    except TypeError:
        points[:, 1] = np.asarray(y)

    try:
        background_color = to_rgba(background_color)
    except:
        background_color = (1, 1, 1, 1)

    background_color_luminance = math.sqrt(
        0.299 * background_color[0] ** 2
        + 0.587 * background_color[1] ** 2
        + 0.114 * background_color[2] ** 2
    )

    # regl-scatterplot currently allows us to use the blue and alpha component
    # of the RGBA to store point data
    for i, x in enumerate(point_data_encodings):
        # The `+2` comes from the fact that the first two components of the RGBA
        # value are reserved for the points' x and y coordinate.
        # The first value describes the component index and the second will be
        # used to store whether the data was already prepared to avoid
        # duplicated work
        point_data_encodings[x] = (i + 2, False)

    ##########
    # Coloring
    if color_norm is not None:
        if callable(color_norm):
            try:
                color_norm.clip = True
            except:
                color_norm = None
        else:
            try:
                vmin, vmax = color_norm
                color_norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
            except:
                color_norm = None

    if color_norm is None:
        color_norm = matplotlib.colors.Normalize(0, 1, clip=True)

    if color_by is not None:
        categories = None
        component_idx, is_component_prepared = point_data_encodings[color_by]
        if not is_component_prepared:
            try:
                if data[color_by].dtype.name == 'category':
                    categories = dict(zip(data[color_by], data[color_by].cat.codes))
                    points[:, component_idx] = data[color_by].cat.codes
                else:
                    points[:, component_idx] = color_norm(data[color_by].values)
            except TypeError:
                points[:, component_idx] = color_norm(np.asarray(color_by))

            # Make sure we don't prepare the data twice
            point_data_encodings[color_by] = (component_idx, True)

        if color_order is not None and categories is not None:
            # Define order of the colors instead of changing `points[:, component_idx]`
            color_order = [categories[cat] for cat in color_order]

        if color_map is not None:
            if categories is None:
                if callable(color_map):
                    # Assuming `color_map` is a Matplotlib LinearSegmentedColormap
                    color_map = color_map(range(256)).tolist()
                elif isinstance(color_map, str):
                    # Assiming `color_map` is the name of a Matplotlib LinearSegmentedColormap
                    color_map = plt.get_cmap(color_map)(range(256)).tolist()
                else:
                    # Assuming `color_map` is a list of colors
                    color_map = [to_rgba(c) for c in color_map]
            else:
                if callable(color_map):
                    # Assuming `color_map` is a Matplotlib ListedColormap
                    color_map = [to_rgba(c) for c in color_map.colors]
                elif isinstance(color_map, str):
                    # Assiming `color_map` is the name of a Matplotlib ListedColormap
                    color_map = [to_rgba(c) for c in plt.get_cmap(color_map).colors]
                else:
                    # Assuming `color_map` is a list of colors
                    color_map = [to_rgba(c) for c in color_map]

                if color_order is not None:
                    color_map = [color_map[color_order[i]] for i, _ in enumerate(color_map)]
        else:
            # Assign default color maps
            if categories is None:
                color_map = plt.get_cmap('viridis')(range(256)).tolist()
            elif len(categories) > 8:
                if background_color_luminance < 0.5:
                    color_map = glasbey_light
                else:
                    color_map = glasbey_dark
            else:
                color_map = okabe_ito

        # Reverse if needed
        color_map = color_map[::(1 + (-2 * (color_order == 'reverse')))]

        if categories is not None:
            assert len(categories) <= len(color_map), 'More categories than colors'

    ##########
    # Opacity
    if opacity_norm is not None:
        if callable(size_norm):
            try:
                opacity_norm.clip = True
            except:
                opacity_norm = None
        else:
            try:
                vmin, vmax = opacity_norm
                opacity_norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
            except:
                opacity_norm = None

    if opacity_norm is None:
        opacity_norm = matplotlib.colors.Normalize(0, 1, clip=True)

    if opacity_by is not None:
        categories = None
        component_idx, is_component_prepared = point_data_encodings[opacity_by]
        if not is_component_prepared:
            try:
                if data[opacity_by].dtype.name == 'category':
                    points[:, component_idx] = data[opacity_by].cat.codes
                    categories = dict(zip(data[opacity_by], data[opacity_by].cat.codes))
                else:
                    points[:, component_idx] = opacity_norm(data[opacity_by].values)
            except TypeError:
                points[:, component_idx] = opacity_norm(np.asarray(opacity_by))

            # Make sure we don't prepare the data twice
            point_data_encodings[opacity_by] = (component_idx, True)

        if opacity_order is not None and categories is not None:
            # Define order of the opacities instead of changing `points[:, component_idx]`
            opacity_order = [categories[cat] for cat in opacity_order]

        if opacity_map is not None:
            if type(opacity_map) == tuple:
                # Assuming `opacity_map` is a triple specifying a linear space
                start, end, num = opacity_map
                opacity_map = np.linspace(start, end, num)
            else:
                opacity_map = np.asarray(opacity_map)

            if categories is not None and size_order is not None:
                opacity_map = np.asarray([opacity_map[size_order[i]] for i, _ in enumerate(opacity_map)])

        else:
            # The best we can do is provide a linear opacity map
            if categories is not None:
                opacity_map = np.linspace(1/len(categories), 1, len(categories))
            else:
                opacity_map = np.linspace(1/256, 1, 256)

        # Reverse if needed
        opacity_map = opacity_map[::(1 + (-2 * (opacity_order == 'reverse')))]
        opacity_map = opacity_map.tolist()


        if categories is not None:
            assert len(categories) <= len(opacity_map), 'More categories than opacities'

    ##########
    # Point Sizing
    if size_norm is not None:
        if callable(size_norm):
            try:
                size_norm.clip = True
            except:
                size_norm = None
        else:
            try:
                vmin, vmax = size_norm
                size_norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
            except:
                size_norm = None

    if size_norm is None:
        size_norm = matplotlib.colors.Normalize(0, 1, clip=True)

    if size_by is not None:
        categories = None
        component_idx, is_component_prepared = point_data_encodings[size_by]
        if not is_component_prepared:
            try:
                if data[size_by].dtype.name == 'category':
                    points[:, component_idx] = data[size_by].cat.codes
                    categories = dict(zip(data[size_by], data[size_by].cat.codes))
                else:
                    points[:, component_idx] = size_norm(data[size_by].values)
            except TypeError:
                points[:, component_idx] = size_norm(np.asarray(size_by))

            # Make sure we don't prepare the data twice
            point_data_encodings[size_by] = (component_idx, True)

        if size_order is not None and categories is not None:
            # Define order of the sizes instead of changing `points[:, component_idx]`
            size_order = [categories[cat] for cat in size_order]

        if size_map is not None:
            if type(size_map) == tuple:
                # Assuming `size_map` is a triple specifying a linear space
                start, end, num = size_map
                size_map = np.linspace(start, end, num)
            else:
                size_map = np.asarray(size_map)

            if categories is not None and size_order is not None:
                size_map = np.asarray([size_map[size_order[i]] for i, _ in enumerate(size_map)])

        else:
            # The best we can do is provide a linear size map
            if categories is None:
                size_map = np.linspace(1, 10, 19)
            else:
                size_map = np.arange(1, len(categories) + 1)

        # Reverse if needed
        size_map = size_map[::(1 + (-2 * (size_order == 'reverse')))]
        size_map = size_map.tolist()

        if categories is not None:
            assert len(categories) <= len(size_map), 'More categories than sizes'

    ##########
    # Point connections
    if connect_by is not None:
        if data[connect_by].dtype.name != 'category':
            raise TypeError('connect_by only works with categorical data')

        categories = None

        try:
            if data[connect_by].dtype.name == 'category':
                points[:, 4] = data[connect_by].cat.codes
                categories = dict(zip(data[connect_by], points[:, 4]))
            else:
                raise TypeError('connect_by only works with categorical data')
        except TypeError:
            tmp = pd.Series(connect_by, dtype='category')
            points[:, 4] = tmp.cat.codes
            categories = dict(zip(tmp, tmp.cat.codes))

        assert categories is not None, 'connect_by data is broken. Do not call the cops but ruuuun!'

        if connect_order is not None:
            # Since regl-scatterplot doesn't support `connect_order` we have to sort the data now
            try:
                # Sort data
                sorting = data.sort_values([connect_by, connect_order]).index.values
                data = data[sorting]
                sort_order = sorting_to_dict(sorting)
            except TypeError:
                raise TypeError('connect_order only works with Pandas data for now')

    ##########
    # Connection Coloring
    if connection_color_norm is not None:
        if callable(connection_color_norm):
            try:
                connection_color_norm.clip = True
            except:
                connection_color_norm = None
        else:
            try:
                vmin, vmax = connection_color_norm
                connection_color_norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
            except:
                connection_color_norm = None

    if connection_color_norm is None:
        connection_color_norm = matplotlib.colors.Normalize(0, 1, clip=True)

    if connection_color_by is not None:
        categories = None
        component_idx, is_component_prepared = point_data_encodings[connection_color_by]
        if not is_component_prepared:
            try:
                if data[connection_color_by].dtype.name == 'category':
                    categories = dict(zip(data[connection_color_by], data[connection_color_by].cat.codes))
                    points[:, component_idx] = data[connection_color_by].cat.codes
                else:
                    points[:, component_idx] = connection_color_norm(data[connection_color_by].values)
            except TypeError:
                points[:, component_idx] = connection_color_norm(np.asarray(connection_color_by))

            # Make sure we don't prepare the data twice
            point_data_encodings[connection_color_by] = (component_idx, True)

        if connection_color_order is not None and categories is not None:
            # Define order of the colors instead of changing `points[:, component_idx]`
            connection_color_order = [categories[cat] for cat in connection_color_order]

        if connection_color_map is not None:
            if categories is None:
                if callable(connection_color_map):
                    # Assuming `connection_color_map` is a Matplotlib LinearSegmentedColormap
                    connection_color_map = connection_color_map(range(256)).tolist()
                elif isinstance(connection_color_map, str):
                    # Assiming `connection_color_map` is the name of a Matplotlib LinearSegmentedColormap
                    connection_color_map = plt.get_cmap(connection_color_map)(range(256)).tolist()
                else:
                    # Assuming `connection_color_map` is a list of colors
                    connection_color_map = [to_rgba(c) for c in connection_color_map]
            else:
                if callable(connection_color_map):
                    # Assuming `connection_color_map` is a Matplotlib ListedColormap
                    connection_color_map = [to_rgba(c) for c in connection_color_map.colors]
                elif isinstance(connection_color_map, str):
                    # Assiming `connection_color_map` is the name of a Matplotlib ListedColormap
                    connection_color_map = [to_rgba(c) for c in plt.get_cmap(connection_color_map).colors]
                else:
                    # Assuming `connection_color_map` is a list of colors
                    connection_color_map = [to_rgba(c) for c in connection_color_map]

                if connection_color_order is not None:
                    connection_color_map = [connection_color_map[connection_color_order[i]] for i, _ in enumerate(connection_color_map)]
        else:
            # Assign default color maps
            if categories is None:
                connection_color_map = plt.get_cmap('viridis')(range(256)).tolist()
            elif len(categories) > 8:
                if background_color_luminance < 0.5:
                    connection_color_map = glasbey_light
                else:
                    connection_color_map = glasbey_dark
            else:
                connection_color_map = okabe_ito

        # Reverse if needed
        connection_color_map = connection_color_map[::(1 + (-2 * (connection_color_order == 'reverse')))]

        if categories is not None:
            assert len(categories) <= len(connection_color_map), 'More categories than connection colors'

    ##########
    # Connection Opacity
    if connection_opacity_norm is not None:
        if callable(size_norm):
            try:
                connection_opacity_norm.clip = True
            except:
                connection_opacity_norm = None
        else:
            try:
                vmin, vmax = connection_opacity_norm
                connection_opacity_norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
            except:
                connection_opacity_norm = None

    if connection_opacity_norm is None:
        connection_opacity_norm = matplotlib.colors.Normalize(0, 1, clip=True)

    if connection_opacity_by is not None:
        categories = None
        component_idx, is_component_prepared = point_data_encodings[connection_opacity_by]
        if not is_component_prepared:
            try:
                if data[connection_opacity_by].dtype.name == 'category':
                    points[:, component_idx] = data[connection_opacity_by].cat.codes
                    categories = dict(zip(data[connection_opacity_by], data[connection_opacity_by].cat.codes))
                else:
                    points[:, component_idx] = connection_opacity_norm(data[connection_opacity_by].values)
            except TypeError:
                points[:, component_idx] = connection_opacity_norm(np.asarray(connection_opacity_by))

            # Make sure we don't prepare the data twice
            point_data_encodings[connection_opacity_by] = (component_idx, True)

        if connection_opacity_order is not None and categories is not None:
            # Define order of the opacities instead of changing `points[:, component_idx]`
            connection_opacity_order = [categories[cat] for cat in connection_opacity_order]

        if connection_opacity_map is not None:
            if type(connection_opacity_map) == tuple:
                # Assuming `connection_opacity_map` is a triple specifying a linear space
                start, end, num = connection_opacity_map
                connection_opacity_map = np.linspace(start, end, num)
            else:
                connection_opacity_map = np.asarray(connection_opacity_map)

            if categories is not None and size_order is not None:
                connection_opacity_map = np.asarray([connection_opacity_map[size_order[i]] for i, _ in enumerate(connection_opacity_map)])

        else:
            # The best we can do is provide a linear opacity map
            if categories is not None:
                connection_opacity_map = np.linspace(1/len(categories), 1, len(categories))
            else:
                connection_opacity_map = np.linspace(1/256, 1, 256)

        # Reverse if needed
        connection_opacity_map = connection_opacity_map[::(1 + (-2 * (connection_opacity_order == 'reverse')))]
        connection_opacity_map = connection_opacity_map.tolist()


        if categories is not None:
            assert len(categories) <= len(connection_opacity_map), 'More categories than connection opacities'

    ##########
    # Point Sizing
    if connection_size_norm is not None:
        if callable(connection_size_norm):
            try:
                connection_size_norm.clip = True
            except:
                connection_size_norm = None
        else:
            try:
                vmin, vmax = connection_size_norm
                connection_size_norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
            except:
                connection_size_norm = None

    if connection_size_norm is None:
        connection_size_norm = matplotlib.colors.Normalize(0, 1, clip=True)

    if connection_size_by is not None:
        categories = None
        component_idx, is_component_prepared = point_data_encodings[connection_size_by]
        if not is_component_prepared:
            try:
                if data[connection_size_by].dtype.name == 'category':
                    points[:, component_idx] = data[connection_size_by].cat.codes
                    categories = dict(zip(data[connection_size_by], data[connection_size_by].cat.codes))
                else:
                    points[:, component_idx] = connection_size_norm(data[connection_size_by].values)
            except TypeError:
                points[:, component_idx] = connection_size_norm(np.asarray(connection_size_by))

            # Make sure we don't prepare the data twice
            point_data_encodings[connection_size_by] = (component_idx, True)

        if connection_size_order is not None and categories is not None:
            # Define order of the sizes instead of changing `points[:, component_idx]`
            connection_size_order = [categories[cat] for cat in connection_size_order]

        if connection_size_map is not None:
            if type(connection_size_map) == tuple:
                # Assuming `connection_size_map` is a triple specifying a linear space
                start, end, num = connection_size_map
                connection_size_map = np.linspace(start, end, num)
            else:
                connection_size_map = np.asarray(connection_size_map)

            if categories is not None and connection_size_order is not None:
                connection_size_map = np.asarray([connection_size_map[connection_size_order[i]] for i, _ in enumerate(connection_size_map)])

        else:
            # The best we can do is provide a linear size map
            if categories is None:
                connection_size_map = np.linspace(1, 10, 19)
            else:
                connection_size_map = np.arange(1, len(categories) + 1)

        # Reverse if needed
        connection_size_map = connection_size_map[::(1 + (-2 * (connection_size_order == 'reverse')))]
        connection_size_map = connection_size_map.tolist()

        if categories is not None:
            assert len(categories) <= len(connection_size_map), 'More categories than connection sizes'

    try:
        color = to_rgba(color)
    except:
        color = (0, 0, 0, 0.66)

    try:
        lasso_color = to_rgba(lasso_color)
    except:
        lasso_color = (1, 1, 1, 1)

    try:
        recticle_color = to_rgba(recticle_color)
    except:
        recticle_color = (1, 1, 1, 1)

    try:
        selection = np.asarray(selection)
    except:
        selection = np.asarray([])

    # Normalize points to [-1,1]
    min_point = points[:, 0:2].min(axis=0)
    max_point = points[:, 0:2].max(axis=0) - min_point
    points[:, 0:2] = (points[:, 0:2] - min_point) / max_point * 2 - 1

    x_domain = np.array(min_point[0], max_point[0])
    y_domain = np.array(min_point[1], max_point[1])

    js_color_by = None
    if color_by is not None:
        js_color_by = component_idx_to_name(point_data_encodings[color_by][0])

    js_opacity_by = None
    if opacity_by is not None:
        js_opacity_by = component_idx_to_name(point_data_encodings[opacity_by][0])

    js_size_by = None
    if size_by is not None:
        js_size_by = component_idx_to_name(point_data_encodings[size_by][0])

    js_connection_color_by = None
    if connection_color_by is not None:
        js_connection_color_by = component_idx_to_name(point_data_encodings[connection_color_by][0])

    js_connection_opacity_by = None
    if connection_opacity_by is not None:
        js_connection_opacity_by = component_idx_to_name(point_data_encodings[connection_opacity_by][0])

    js_connection_size_by = None
    if connection_size_by is not None:
        js_connection_size_by = component_idx_to_name(point_data_encodings[connection_size_by][0])

    return JupyterScatter(
        points=points.tolist(),
        selected_points=selection.tolist(),
        height=height,
        background_color=background_color,
        background_image=background_image,
        lasso_color=lasso_color,
        lasso_min_delay=lasso_min_delay,
        lasso_min_dist=lasso_min_dist,
        color=color_map or color,
        color_active=color_active,
        color_hover=color_hover,
        color_by=js_color_by,
        opacity=opacity_map or opacity,
        opacity_by=js_opacity_by,
        size=size_map or size,
        size_by=js_size_by,
        connect=bool(connect_by),
        connection_color=connection_color_map or connection_color,
        connection_color_active=connection_color_active,
        connection_color_hover=connection_color_hover,
        connection_color_by=js_connection_color_by,
        connection_opacity=connection_opacity_map or connection_opacity,
        connection_opacity_by=js_connection_opacity_by,
        connection_size=connection_size_map or connection_size,
        connection_size_by=js_connection_size_by,
        selection_size_addition=selection_size_addition,
        selection_outline_width=selection_outline_width,
        recticle=recticle,
        recticle_color=recticle_color,
        camera_target=camera_target,
        camera_distance=camera_distance,
        camera_rotation=camera_rotation,
        camera_view=camera_view,
        mouse_mode=mouse_mode,
        lasso_initiator=lasso_initiator,
        x_domain=x_domain.tolist(),
        y_domain=y_domain.tolist(),
        other_options=options,
        sort_order=sort_order
    )
