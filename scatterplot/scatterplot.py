import base64
import IPython.display as ipydisplay
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from traitlets import Bool, Dict, Enum, Float, Int, List, Unicode, Union

from .__version__ import __version__
from .utils import to_hex, with_left_label


@widgets.register
class Scatterplot(widgets.DOMWidget):
    _view_name = Unicode("ScatterplotView").tag(sync=True)
    _model_name = Unicode("ScatterplotModel").tag(sync=True)
    _view_module = Unicode("jupyter-scatterplot").tag(sync=True)
    _model_module = Unicode("jupyter-scatterplot").tag(sync=True)
    _view_module_version = Unicode(__version__).tag(sync=True)
    _model_module_version = Unicode(__version__).tag(sync=True)
    _model_data = List([]).tag(sync=True)

    # For debugging
    dom_element_id = Unicode(read_only=True).tag(sync=True)

    # Data
    points = List().tag(sync=True)
    selected_points = List().tag(sync=True)
    hovered_point = Int(None, allow_none=True, read_only=True).tag(sync=True)

    # View properties
    camera_target = List().tag(sync=True)
    camera_distance = Float().tag(sync=True)
    camera_rotation = Float().tag(sync=True)
    camera_view = List(None, allow_none=True).tag(sync=True)

    # Options
    color_by = Enum([None, 'category', 'value'], allow_none=True, default_value=None).tag(sync=True)
    height = Int().tag(sync=True)
    background_color = Union([Unicode(), List(minlen=4, maxlen=4)]).tag(sync=True)
    background_image = Unicode(None, allow_none=True).tag(sync=True)
    lasso_color = Union([Unicode(), List(minlen=4, maxlen=4)]).tag(sync=True)
    lasso_min_delay = Int().tag(sync=True)
    lasso_min_dist = Float().tag(sync=True)
    point_color = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    point_color_active = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    point_color_hover = Union([Union([Unicode(), List(minlen=4, maxlen=4)]), List(Union([Unicode(), List(minlen=4, maxlen=4)]))]).tag(sync=True)
    point_opacity = Float().tag(sync=True)
    point_size = Int().tag(sync=True)
    point_size_selected = Int().tag(sync=True)
    point_outline_width = Int().tag(sync=True)
    show_recticle = Bool().tag(sync=True)
    recticle_color = Union([Unicode(), List(minlen=4, maxlen=4)]).tag(sync=True)

    # For any kind of options. Note that whatever is defined in options will
    # be overwritten by the short-hand options
    options = Dict({}).tag(sync=True)

    @property
    def selected_points_widget(self):
        widget = widgets.Textarea(
            value=', '.join(f'{n}' for n in self.selected_points),
            placeholder='The indices of selected points will be shown here',
            disabled=True
        )

        def change_handler(change):
            widget.value = ', '.join(f'{n}' for n in change.new)

        self.observe(change_handler, names='selected_points')

        return with_left_label('Selected points', widget)

    @property
    def hovered_point_widget(self):
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
            value='none'
        )

        colormap_widget = widgets.Combobox(
            placeholder='Choose a Matplotlib colormap',
            options=list(plt.colormaps()),
            ensure_option=True,
            disabled=True,
        )

        point_color_widget = widgets.ColorPicker(
            value=to_hex(self.point_color),
            disabled=False
        )

        def color_by_change_handler(change):
            if change.new == 'none':
                self.color_by = None
                self.point_color = point_color_widget.value

                colormap_widget.disabled = True
                point_color_widget.disabled = False

            else:
                self.color_by = change.new
                self.use_cmap(colormap_widget.value)

                colormap_widget.disabled = False
                point_color_widget.disabled = True

        def colormap_change_handler(change):
            self.use_cmap(change.new)

        def point_color_change_handler(change):
            self.point_color = change.new

        color_by_widget.observe(color_by_change_handler, names='value')
        colormap_widget.observe(colormap_change_handler, names='value')
        point_color_widget.observe(point_color_change_handler, names='value')

        return (
            with_left_label('Color by', color_by_widget),
            with_left_label('Colormap', colormap_widget),
            with_left_label('Point color', point_color_widget)
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
    def point_color_widget(self):
        widget = widgets.ColorPicker(value=to_hex(self.point_color))

        def change_handler(change):
            self.point_color = change.new

        widget.observe(change_handler, names='value')

        return with_left_label('Point color', widget)

    @property
    def point_color_active_widget(self):
        widget = widgets.ColorPicker(value=to_hex(self.point_color_active))

        def change_handler(change):
            self.point_color_active = change.new

        widget.observe(change_handler, names='value')

        return with_left_label('Point color active', widget)

    @property
    def point_color_hover_widget(self):
        widget = widgets.ColorPicker(value=to_hex(self.point_color_hover))

        def change_handler(change):
            self.point_color_hover = change.new

        widget.observe(change_handler, names='value')

        return with_left_label('Point color hover', widget)

    @property
    def point_opacity_widget(self):
        widget = widgets.FloatSlider(
            value=self.point_opacity,
            min=0.0,
            max=1.0,
            step=0.05,
            continuous_update=True
        )
        widgets.jslink((self, 'point_opacity'), (widget, 'value'))
        return with_left_label('Point opacity', widget)

    @property
    def point_outline_width_widget(self):
        widget = widgets.IntSlider(
            value=self.point_outline_width,
            min=0,
            max=max(8, self.point_outline_width + 2),
            continuous_update=True
        )
        widgets.jslink((self, 'point_outline_width'), (widget, 'value'))
        return with_left_label('Point outline width', widget)

    @property
    def point_size_widget(self):
        widget = widgets.IntSlider(
            value=self.point_size,
            min=1,
            max=max(10, self.point_size + 5),
            continuous_update=True
        )
        widgets.jslink((self, 'point_size'), (widget, 'value'))
        return with_left_label('Point size', widget)

    @property
    def point_size_selected_widget(self):
        widget = widgets.IntSlider(
            value=self.point_size_selected,
            min=0,
            max=max(8, self.point_size_selected + 2),
            continuous_update=True
        )
        widgets.jslink((self, 'point_size_selected'), (widget, 'value'))
        return with_left_label('Point size selected', widget)

    @property
    def show_recticle_widget(self):
        widget = widgets.Checkbox(
            icon='check',
            indent=False,
            value=self.show_recticle
        )
        widgets.jslink((self, 'show_recticle'), (widget, 'value'))
        return with_left_label('Show recticle', widget)

    @property
    def recticle_color_widget(self):
        widget = widgets.ColorPicker(value=to_hex(self.recticle_color))

        def change_handler(change):
            self.recticle_color = change.new

        widget.observe(change_handler, names='value')

        return with_left_label('Recticle color', widget)

    def display_options(self):
        color_by_widget, colormap_widget, point_color_widget = self.color_widgets
        ipydisplay.display(
            self.height_widget,
            self.point_size_widget,
            self.point_opacity_widget,
            color_by_widget,
            colormap_widget,
            point_color_widget,
            self.point_color_active_widget,
            self.point_color_hover_widget,
            self.point_outline_width_widget,
            self.point_size_selected_widget,
            self.lasso_color_widget,
            self.recticle_color_widget,
            self.show_recticle_widget,
            self.background_color_widget,
            self.background_image_widget,
        )

    def use_cmap(self, cmap_name, reverse=False):
        """
        Create a compatible colormap from a matplotlib colormap.
        Parameters
        ----------
        cmap_name (string):
            The name of the color map.
        reverse (boolean):
            Reverse the colormap if ``True``.
        """
        self.point_color = plt.get_cmap(cmap_name)(range(256)).tolist()[::(1 + (-2 * reverse))]


def display(
    points,
    categories: list = None,
    values: list = None,
    color_by: str = None,
    selected_points: list = [],
    height: int = 240,
    background_color: list = [1, 1, 1, 1],
    background_image: str = None,
    lasso_color: list = [0, 0, 0, 1],
    lasso_min_delay: int = 10,
    lasso_min_dist: float = 3.0,
    point_color: list = [0.66, 0.66, 0.66, 1],
    point_color_active: list = [0, 0.55, 1, 1],
    point_color_hover: list = [0, 0, 0, 1],
    point_opacity: float = 1.0,
    point_size: int = 4,
    point_size_selected: int = 2,
    point_outline_width: int = 2,
    show_recticle: bool = False,
    recticle_color: list = [0, 0.55, 1, 0.33],
    camera_target: list = [0, 0],
    camera_distance: float = 1.0,
    camera_rotation: float = 0.0,
    camera_view: list = None,
    options: dict = {},
):
    points = np.asarray(points)
    n = points.shape[0]

    if points.ndim != 2:
        raise ValueError('The points array must be matrix')

    if points.shape[1] != 2:
        raise ValueError('Only 2D point data is supported')

    min_point = points.min(axis=0)
    max_point = points.max(axis=0) - min_point
    points = (points - min_point) / max_point * 2 - 1

    data = np.zeros((n, 4))
    data[:,:2] = points

    if categories is not None:
        categories = np.asarray(categories)

        if categories.ndim != 1:
            raise ValueError('The categories array must be vector')

        if categories.shape[0] != n:
            raise ValueError('The categories array must be of the same size as the points array')

        data[:,2] = categories

    if values is not None:
        values = np.asarray(values)

        if values.ndim != 1:
            raise ValueError('The value array must be vector')

        if values.shape[0] != n:
            raise ValueError('The value array must be of the same size as the points array')

        data[:,3] = values

    return Scatterplot(
        points=data.tolist(),
        selected_points=selected_points,
        height=height,
        background_color=background_color,
        background_image=background_image,
        lasso_color=lasso_color,
        lasso_min_delay=lasso_min_delay,
        lasso_min_dist=lasso_min_dist,
        point_color=point_color,
        point_color_active=point_color_active,
        point_color_hover=point_color_hover,
        point_opacity=point_opacity,
        point_size=point_size,
        point_size_selected=point_size_selected,
        point_outline_width=point_outline_width,
        show_recticle=show_recticle,
        recticle_color=recticle_color,
        camera_target=camera_target,
        camera_distance=camera_distance,
        camera_rotation=camera_rotation,
        camera_view=camera_view,
        options=options,
    )
