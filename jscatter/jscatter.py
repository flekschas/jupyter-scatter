import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

from matplotlib.colors import to_rgba

from .encodings import Encodings
from .widget import JupyterScatter, SELECTION_DTYPE
from .color_maps import okabe_ito, glasbey_light, glasbey_dark
from .utils import any_not, minmax_scale, tolist, uri_validator

COMPONENT_CONNECT = 4
COMPONENT_CONNECT_ORDER = 5
VALID_ENCODING_TYPES = [
    pd.api.types.is_float_dtype,
    pd.api.types.is_integer_dtype,
    pd.api.types.is_categorical_dtype,
]

# To distinguish between None and an undefined optional argument
Undefined = object()

default_norm = matplotlib.colors.Normalize(0, 1, clip=True)
default_background_color = 'white'

def check_encoding_dtype(series):
    if not any([check(series.dtype) for check in VALID_ENCODING_TYPES]):
        raise ValueError(f'{series.name} is of an unsupported data type: {series.dtype}. Must be one of category, float*, or int*.')

def component_idx_to_name(idx):
    if idx == 2:
        return 'valueA'

    if idx == 3:
        return 'valueB'

    return None

def order_map(map, order):
    ordered_map = map
    try:
        ordered_map = [ordered_map[order[i]] for i, _ in enumerate(ordered_map)]
    except TypeError:
        pass

    return ordered_map[::(1 + (-2 * (order == 'reverse')))]


class Scatter():
    def __init__(self, x, y, data = None, **kwargs):
        self._data = data

        try:
            self._n = len(self._data)
        except TypeError:
            self._n = np.asarray(x).size

        self._points = np.zeros((self._n, 6))
        self._widget = None
        self._pixels = None
        self._encodings = Encodings()
        self._selection = np.asarray([], dtype=SELECTION_DTYPE)
        self._background_color = to_rgba(default_background_color)
        self._background_color_luminance = 1
        self._background_image = None
        self._lasso_color = (0, 0.666666667, 1, 1)
        self._lasso_initiator = True
        self._lasso_min_delay = 10
        self._lasso_min_dist = 3
        self._color = (0, 0, 0, 0.66)
        self._color_active = (0, 0.55, 1, 1)
        self._color_hover = (0, 0, 0, 1)
        self._color_by = None
        self._color_map = None
        self._color_norm = default_norm
        self._color_order = None
        self._color_categories = None
        self._opacity = 0.66
        self._opacity_by = 'density'
        self._opacity_map = None
        self._opacity_norm = default_norm
        self._opacity_order = None
        self._opacity_categories = None
        self._size = 3
        self._size_by = None
        self._size_map = None
        self._size_norm = default_norm
        self._size_order = None
        self._size_categories = None
        self._connect_by = None
        self._connect_order = None
        self._connection_color = (0, 0, 0, 0.1)
        self._connection_color_active = (0, 0.55, 1, 1)
        self._connection_color_hover = (0, 0, 0, 0.66)
        self._connection_color_by = None
        self._connection_color_map = None
        self._connection_color_norm = default_norm
        self._connection_color_order = None
        self._connection_color_categories = None
        self._connection_opacity = 0.1
        self._connection_opacity_by = None
        self._connection_opacity_map = None
        self._connection_opacity_norm = default_norm
        self._connection_opacity_order = None
        self._connection_opacity_categories = None
        self._connection_size = 2
        self._connection_size_by = None
        self._connection_size_map = None
        self._connection_size_norm = default_norm
        self._connection_size_order = None
        self._connection_size_categories = None
        self._width = 'auto'
        self._height = 240
        self._reticle = True
        self._reticle_color = 'auto'
        self._camera_target = [0, 0]
        self._camera_distance = 1.0
        self._camera_rotation = 0.0
        self._camera_view = None
        self._mouse_mode = 'panZoom'
        self._options = {}

        self.x(x)
        self.y(y)
        self.width(kwargs.get('width', Undefined))
        self.height(kwargs.get('height', Undefined))
        self.selection(
            kwargs.get('selection', Undefined),
        )
        self.color(
            kwargs.get('color', Undefined),
            kwargs.get('color_active', Undefined),
            kwargs.get('color_hover', Undefined),
            kwargs.get('color_by', Undefined),
            kwargs.get('color_map', Undefined),
            kwargs.get('color_norm', Undefined),
            kwargs.get('color_order', Undefined),
        )
        self.opacity(
            kwargs.get('opacity', Undefined),
            kwargs.get('opacity_by', Undefined),
            kwargs.get('opacity_map', Undefined),
            kwargs.get('opacity_norm', Undefined),
            kwargs.get('opacity_order', Undefined),
        )
        self.size(
            kwargs.get('size', Undefined),
            kwargs.get('size_by', Undefined),
            kwargs.get('size_map', Undefined),
            kwargs.get('size_norm', Undefined),
            kwargs.get('size_order', Undefined),
        )
        self.connect(
            kwargs.get('connect_by', Undefined),
            kwargs.get('connect_order', Undefined)
        )
        self.connection_color(
            kwargs.get('connection_color', Undefined),
            kwargs.get('connection_color_active', Undefined),
            kwargs.get('connection_color_hover', Undefined),
            kwargs.get('connection_color_by', Undefined),
            kwargs.get('connection_color_map', Undefined),
            kwargs.get('connection_color_norm', Undefined),
            kwargs.get('connection_color_order', Undefined),
        )
        self.connection_opacity(
            kwargs.get('connection_opacity', Undefined),
            kwargs.get('connection_opacity_by', Undefined),
            kwargs.get('connection_opacity_map', Undefined),
            kwargs.get('connection_opacity_norm', Undefined),
            kwargs.get('connection_opacity_order', Undefined),
        )
        self.connection_size(
            kwargs.get('connection_size', Undefined),
            kwargs.get('connection_size_by', Undefined),
            kwargs.get('connection_size_map', Undefined),
            kwargs.get('connection_size_order', Undefined),
        )
        self.lasso(
            kwargs.get('lasso_color', Undefined),
            kwargs.get('lasso_initiator', Undefined),
            kwargs.get('lasso_min_delay', Undefined),
            kwargs.get('lasso_min_dist', Undefined),
        )
        self.reticle(
            kwargs.get('reticle', Undefined),
            kwargs.get('reticle_color', Undefined)
        )
        self.background(
            kwargs.get('background_color', Undefined),
            kwargs.get('background_image', Undefined),
        )
        self.mouse(kwargs.get('mouse_mode', Undefined))
        self.camera(
            kwargs.get('camera_target', Undefined),
            kwargs.get('camera_distance', Undefined),
            kwargs.get('camera_rotation', Undefined),
            kwargs.get('camera_view', Undefined),
        )
        self.options(kwargs.get('options', Undefined))

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

    def x(self, x = Undefined, **kwargs):
        if x is not Undefined:
            self._x = x

            try:
                self._points[:, 0] = self._data[x].values
            except TypeError:
                self._points[:, 0] = np.asarray(x)

            # Normalize x coordinate to [-1,1]
            self._x_min = np.min(self._points[:, 0])
            self._x_max = np.max(self._points[:, 0])
            self._x_domain = [self._x_min, self._x_max]

            self._points[:, 0] = minmax_scale(self._points[:, 0], (-1,1))

            if 'skip_widget_update' not in kwargs:
                self.update_widget('points', self.get_point_list())

            return self

        return self._x

    def y(self, y = Undefined, **kwargs):
        if y is not Undefined:
            self._y = y

            try:
                self._points[:, 1] = self._data[y].values
            except TypeError:
                self._points[:, 1] = np.asarray(y)

            # Normalize y coordinate to [-1,1]
            self._y_min = np.min(self._points[:, 1])
            self._y_max = np.max(self._points[:, 1])
            self._y_domain = [self._y_min, self._y_max]

            self._points[:, 1] = minmax_scale(self._points[:, 1], (-1,1))

            if 'skip_widget_update' not in kwargs:
                self.update_widget('points', self.get_point_list())

            return self

        return self._y

    def xy(self, x, y, **kwargs):
        self.x(x, skip_widget_update=True)
        self.y(y, skip_widget_update=True)

        if 'skip_widget_update' not in kwargs:
            self.update_widget('points', self.get_point_list())

    def selection(self, selection = Undefined):
        if selection is not Undefined:
            try:
                self._selection = np.asarray(selection).astype(SELECTION_DTYPE)
                self.update_widget('selection', self._selection)
            except:
                if selection is None:
                    self._selection = np.asarray([], dtype=SELECTION_DTYPE)
                pass

            return self

        if self._widget is not None:
            return self._widget.selection.astype(SELECTION_DTYPE)

        return self._selection

    def color(
        self,
        color = Undefined,
        color_active = Undefined,
        color_hover = Undefined,
        by = Undefined,
        map = Undefined,
        norm = Undefined,
        order = Undefined,
        **kwargs
    ):
        if color is not Undefined:
            try:
                self._color = to_rgba(color)
            except ValueError:
                pass

        if color_active is not Undefined:
            try:
                self._color_active = to_rgba(color_active)
                self.update_widget('color_active', self._color_active)
            except ValueError:
                pass

        if color_hover is not Undefined:
            try:
                self._color_hover = to_rgba(color_hover)
                self.update_widget('color_hover', self._color_hover)
            except ValueError:
                pass

        if norm is not Undefined:
            if callable(norm):
                try:
                    self._color_norm = norm
                    self._color_norm.clip = True
                except:
                    pass
            else:
                try:
                    vmin, vmax = norm
                    self._color_norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
                except:
                    if norm is None:
                        self._color_norm = default_norm
                    pass

        data_updated = False
        if by is not Undefined:
            self._color_by = by

            if by is None:
                self._encodings.delete('color')

            else:
                self._encodings.set('color', by)

                if not self._encodings.data[by].prepared:
                    component = self._encodings.data[by].component
                    try:
                        check_encoding_dtype(self._data[by])
                        if pd.api.types.is_categorical_dtype(self._data[by].dtype):
                            self._color_categories = dict(zip(self._data[by], self._data[by].cat.codes))
                            self._points[:, component] = self._data[by].cat.codes
                        else:
                            self._points[:, component] = self._color_norm(self._data[by].values)
                    except TypeError:
                        self._points[:, component] = self._color_norm(np.asarray(by))

                    data_updated = True

                    # Make sure we don't prepare the data twice
                    self._encodings.data[by].prepared = True

            self.update_widget('color_by', self.js_color_by)

        if order is not Undefined:
            if order in [None, 'reverse']:
                self._color_order = order
            elif self._color_categories is not None:
                # Define order of the colors instead of changing `points[:, component_idx]`
                self._color_order = [self._color_categories[cat] for cat in order]

        if map is not Undefined and map != 'auto':
            if self._color_categories is None:
                if callable(map):
                    # Assuming `map` is a Matplotlib LinearSegmentedColormap
                    self._color_map = map(range(256)).tolist()
                elif isinstance(map, str):
                    # Assiming `map` is the name of a Matplotlib LinearSegmentedColormap
                    self._color_map = plt.get_cmap(map)(range(256)).tolist()
                else:
                    # Assuming `map` is a list of colors
                    self._color_map = [to_rgba(c) for c in map]
            else:
                if callable(map):
                    # Assuming `map` is a Matplotlib ListedColormap
                    self._color_map = [to_rgba(c) for c in map.colors]
                elif isinstance(map, str):
                    # Assiming `map` is the name of a Matplotlib ListedColormap
                    self._color_map = [to_rgba(c) for c in plt.get_cmap(map).colors]
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

        # Update widget
        if self._color_by is not None and self._color_map is not None:
            self.update_widget(
                'color',
                order_map(self._color_map, self._color_order)
            )
        else:
            self.update_widget('color', self._color)

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('points', self.get_point_list())

        if any_not([color, color_active, color_hover, by, map, norm, order], Undefined):
            return self

        return dict(
            color = self._color,
            color_active = self._color_active,
            color_hover = self._color_hover,
            by = self._color_by,
            map = self._color_map,
            norm = self._color_norm,
            order = self._color_order,
        )

    def opacity(
        self,
        opacity = Undefined,
        by = Undefined,
        map = Undefined,
        norm = Undefined,
        order = Undefined,
        **kwargs
    ):
        if opacity is not Undefined:
            try:
                self._opacity = float(opacity)
                assert self._opacity >= 0 and self._opacity <= 1, 'Opacity must be in [0,1]'
            except ValueError:
                pass

        if norm is not Undefined:
            if callable(norm):
                try:
                    self._opacity_norm = norm
                    self._opacity_norm.clip = True
                except:
                    pass
            else:
                try:
                    vmin, vmax = norm
                    self._opacity_norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
                except:
                    if norm is None:
                        # Reset to default value
                        self._opacity_norm = default_norm
                    pass

        data_updated = False
        if by is not Undefined:
            self._opacity_by = by

            if by is None:
                self._encodings.delete('opacity')

            elif by == 'density':
                pass

            else:
                self._encodings.set('opacity', by)

                if not self._encodings.data[by].prepared:
                    component = self._encodings.data[by].component
                    try:
                        check_encoding_dtype(self._data[by])
                        if pd.api.types.is_categorical_dtype(self._data[by].dtype):
                            self._points[:, component] = self._data[by].cat.codes
                            self._opacity_categories = dict(zip(self._data[by], self._data[by].cat.codes))
                        else:
                            self._points[:, component] = self._opacity_norm(self._data[by].values)
                    except TypeError:
                        self._points[:, component] = self._opacity_norm(np.asarray(by))

                    data_updated = True

                    # Make sure we don't prepare the data twice
                    self._encodings.data[by].prepared = True

            self.update_widget('opacity_by', self.js_opacity_by)

        if order is not Undefined:
            if order in [None, 'reverse']:
                self._opacity_order = order
            elif self._opacity_categories is not None:
                # Define order of the opacities instead of changing `points[:, component_idx]`
                self._opacity_order = [self._opacity_categories[cat] for cat in order]

        if map is not Undefined and map != 'auto':
            if type(map) == tuple:
                # Assuming `map` is a triple specifying a linear space
                self._opacity_map = np.linspace(*map)
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

        # Update widget
        if self._opacity_by is not None and self._opacity_map is not None:
            self.update_widget(
                'opacity',
                order_map(self._opacity_map, self._opacity_order)
            )
        else:
            self.update_widget('opacity', self._opacity)

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('points', self.get_point_list())

        if any_not([opacity, by, map, norm, order], Undefined):
            return self

        return dict(
            opacity = self._opacity,
            by = self._opacity_by,
            map = self._opacity_map,
            norm = self._opacity_norm,
            order = self._opacity_order,
        )

    def size(
        self,
        size = Undefined,
        by = Undefined,
        map = Undefined,
        norm = Undefined,
        order = Undefined,
        **kwargs
    ):
        if size is not Undefined:
            try:
                self._size = int(size)
                assert self._size > 0, 'Size must be a positive integer'
                self.update_widget('size', self._size_map or self._size)
            except ValueError:
                pass

        if norm is not Undefined:
            if callable(norm):
                try:
                    self._size_norm = norm
                    self._size_norm.clip = True
                except:
                    pass
            else:
                try:
                    vmin, vmax = norm
                    self._size_norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
                except:
                    if norm is not None:
                        self._size_norm = default_norm
                    pass

        data_updated = False
        if by is not Undefined:
            self._size_by = by

            if by is None:
                self._encodings.delete('size')

            else:
                self._encodings.set('size', by)

                if not self._encodings.data[by].prepared:
                    component = self._encodings.data[by].component
                    try:
                        check_encoding_dtype(self._data[by])
                        if pd.api.types.is_categorical_dtype(self._data[by].dtype):
                            self._points[:, component] = self._data[by].cat.codes
                            self._size_categories = dict(zip(self._data[by], self._data[by].cat.codes))
                        else:
                            self._points[:, component] = self._size_norm(self._data[by].values)
                    except TypeError:
                        self._points[:, component] = self._size_norm(np.asarray(by))

                    data_updated = True

                    # Make sure we don't prepare the data twice
                    self._encodings.data[by].prepared = True

            self.update_widget('size_by', self.js_size_by)

        if order is not Undefined:
            if order in [None, 'reverse']:
                self._size_order = order
            elif self._size_categories is not None:
                # Define order of the sizes instead of changing `points[:, component_idx]`
                self._size_order = [self._size_categories[cat] for cat in self._size_order]

        if map is not Undefined and map != 'auto':
            if type(map) == tuple:
                # Assuming `map` is a triple specifying a linear space
                self._size_map = np.linspace(*map)
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

        # Update widget
        if self._size_by is not None and self._size_map is not None:
            self.update_widget(
                'size', order_map(self._size_map, self._size_order)
            )
        else:
            self.update_widget('size', self._size)

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('points', self.get_point_list())

        if any_not([size, by, map, norm, order], Undefined):
            return self

        return dict(
            size = self._size,
            by = self._size_by,
            map = self._size_map,
            norm = self._size_norm,
            order = self._size_order,
        )

    def connect(self, by = Undefined, order = Undefined, **kwargs):
        data_updated = False
        if by is not Undefined:
            self._connect_by = by

            if by is not None:
                try:
                    if pd.api.types.is_categorical_dtype(self._data[by].dtype):
                        self._points[:, COMPONENT_CONNECT] = self._data[by].cat.codes
                    elif pd.api.types.is_integer_dtype(self._data[by].dtype):
                        self._points[:, COMPONENT_CONNECT] = self._data[by].values
                    else:
                        raise ValueError('connect by only works with categorical data')
                except TypeError:
                    tmp = pd.Series(by, dtype='category')
                    self._points[:, COMPONENT_CONNECT] = tmp.cat.codes

                data_updated = True

        if order is not Undefined:
            self._connect_order = order

            try:
                if pd.api.types.is_integer_dtype(self._data[order].dtype):
                    self._points[:, COMPONENT_CONNECT_ORDER] = self._data[order]
                else:
                    raise ValueError('connect order must be an integer type')
            except TypeError:
                self._points[:, COMPONENT_CONNECT_ORDER] = np.asarray(order).astype(int)

            data_updated = True

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('points', self.get_point_list())

        self.update_widget('connect', bool(self._connect_by))

        if any_not([by, order], Undefined):
            return self

        return dict(
            by = self._connect_by,
            order = self._connect_order,
        )

    def connection_color(
        self,
        color = Undefined,
        color_active = Undefined,
        color_hover = Undefined,
        by = Undefined,
        map = Undefined,
        norm = Undefined,
        order = Undefined,
        **kwargs
    ):
        if color is not Undefined:
            try:
                self._connection_color = to_rgba(color)
            except ValueError:
                pass

        if color_active is not Undefined:
            try:
                self._connection_color_active = to_rgba(color_active)
                self.update_widget('connection_color_active', self._connection_color_active)
            except ValueError:
                pass

        if color_hover is not Undefined:
            try:
                self._connection_color_hover = to_rgba(color_hover)
                self.update_widget('connection_color_hover', self._connection_color_hover)
            except ValueError:
                pass

        if norm is not Undefined:
            if callable(norm):
                try:
                    self._connection_color_norm = norm
                    self._connection_color_norm.clip = True
                except:
                    pass
            else:
                try:
                    vmin, vmax = norm
                    self._connection_color_norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
                except:
                    if norm is None:
                        self._connection_color_norm = default_norm
                    pass

        data_updated = True
        if by is not Undefined:
            self._connection_color_by = by

            if by is None:
                self._encodings.delete('connection_color')

            elif by == 'segment':
                pass

            else:
                self._encodings.set('connection_color', by)

                if not self._encodings.data[by].prepared:
                    component = self._encodings.data[by].component
                    try:
                        check_encoding_dtype(self._data[by])
                        if pd.api.types.is_categorical_dtype(self._data[by].dtype):
                            self._connection_color_categories = dict(zip(self._data[by], self._data[by].cat.codes))
                            self._points[:, component] = self._data[by].cat.codes
                        else:
                            self._points[:, component] = self._connection_color_norm(self._data[by].values)
                    except TypeError:
                        self._points[:, component] = self._connection_color_norm(np.asarray(by))

                    data_updated = True

                    # Make sure we don't prepare the data twice
                    self._encodings.data[by].prepared = True

            self.update_widget('connection_color_by', self.js_connection_color_by)

        if order is not Undefined:
            if order in [None, 'reverse']:
                self._connection_color_order = order
            elif self._connection_color_categories is not None:
                # Define order of the colors instead of changing `points[:, component_idx]`
                self._connection_color_order = [self._connection_color_categories[cat] for cat in order]

        if map is not Undefined and map != 'auto':
            if self._connection_color_categories is None:
                if callable(map):
                    # Assuming `map` is a Matplotlib LinearSegmentedColormap
                    self._connection_color_map = map(range(256)).tolist()
                elif isinstance(map, str):
                    # Assiming `map` is the name of a Matplotlib LinearSegmentedColormap
                    self._connection_color_map = plt.get_cmap(map)(range(256)).tolist()
                else:
                    # Assuming `map` is a list of colors
                    self._connection_color_map = [to_rgba(c) for c in map]
            else:
                if callable(map):
                    # Assuming `map` is a Matplotlib ListedColormap
                    self._connection_color_map = [to_rgba(c) for c in map.colors]
                elif isinstance(map, str):
                    # Assiming `map` is the name of a Matplotlib ListedColormap
                    self._connection_color_map = [to_rgba(c) for c in plt.get_cmap(map).colors]
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

        # Update widget
        if self._connection_color_by is not None and self._connection_color_map is not None:
            self.update_widget(
                'connection_color',
                order_map(self._connection_color_map, self._connection_color_order)
            )
        else:
            self.update_widget('connection_color', self._connection_color)

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('points', self.get_point_list())

        if any_not([color, by, map, norm, order], Undefined):
            return self

        return dict(
            color = self._connection_color,
            by = self._connection_color_by,
            map = self._connection_color_map,
            norm = self._connection_color_norm,
            order = self._connection_color_order,
        )

    def connection_opacity(
        self,
        opacity = Undefined,
        by = Undefined,
        map = Undefined,
        norm = Undefined,
        order = Undefined,
        **kwargs
    ):
        if opacity is not Undefined:
            try:
                self._connection_opacity = float(opacity)
                assert self._connection_opacity >= 0 and self._connection_opacity <= 1, 'Connection opacity must be in [0,1]'
                self.update_widget('connection_opacity', self._connection_opacity_map or self._connection_opacity)
            except ValueError:
                pass

        if norm is not Undefined:
            if callable(norm):
                try:
                    self._connection_opacity_norm.clip = norm
                    self._connection_opacity_norm.clip = True
                except:
                    pass
            else:
                try:
                    vmin, vmax = norm
                    self._connection_opacity_norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
                except:
                    if norm is None:
                        self._connection_opacity_norm = default_norm
                    pass

        data_updated = False
        if by is not Undefined:
            self._connection_opacity_by = by

            if by is None:
                self._encodings.delete('connection_opacity')

            else:
                self._encodings.set('connection_opacity', by)

                if not self._encodings.data[by].prepared:
                    component = self._encodings.data[by].component
                    try:
                        check_encoding_dtype(self._data[by])
                        if pd.api.types.is_categorical_dtype(self._data[by].dtype):
                            self._points[:, component] = self._data[by].cat.codes
                            self._connection_opacity_categories = dict(zip(self._data[by], self._data[by].cat.codes))
                        else:
                            self._points[:, component] = self._connection_opacity_norm(self._data[by].values)
                    except TypeError:
                        self._points[:, component] = self._connection_opacity_norm(np.asarray(by))

                    data_updated = True

                    # Make sure we don't prepare the data twice
                    self._encodings.data[by].prepared = True

            self.update_widget('connection_opacity_by', self.js_connection_opacity_by)

        if order is not Undefined:
            if order in [None, 'reverse']:
                self._connection_opacity_order = order
            elif self._connection_opacity_categories is not None:
                # Define order of the opacities instead of changing `points[:, component_idx]`
                self._connection_opacity_order = [
                    self._connection_opacity_categories[cat] for cat in order
                ]

        if map is not Undefined and map != 'auto':
            if type(map) == tuple:
                # Assuming `map` is a triple specifying a linear space
                self._connection_opacity_map = np.linspace(*map)
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

        # Update widget
        if self._connection_opacity_by is not None and self._connection_opacity_map is not None:
            self.update_widget(
                'connection_opacity',
                order_map(self._connection_opacity_map, self._connection_opacity_order)
            )
        else:
            self.update_widget('connection_opacity', self._connection_opacity)

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('points', self.get_point_list())

        if any_not([opacity, by, map, norm, order], Undefined):
            return self

        return dict(
            opacity = self._connection_opacity,
            by = self._connection_opacity_by,
            map = self._connection_opacity_map,
            norm = self._connection_opacity_norm,
            order = self._connection_opacity_order,
        )

    def connection_size(
        self,
        size = Undefined,
        by = Undefined,
        map = Undefined,
        norm = Undefined,
        order = Undefined,
        **kwargs
    ):
        if size is not Undefined:
            try:
                self._connection_size = int(size)
                assert self._connection_size > 0, 'Connection size must be a positive integer'
            except ValueError:
                pass

        if norm is not Undefined:
            if callable(norm):
                try:
                    self._connection_size_norm = norm
                    self._connection_size_norm.clip = True
                except:
                    pass
            else:
                try:
                    vmin, vmax = norm
                    self._connection_size_norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
                except:
                    if norm is None:
                        self._connection_size_norm = default_norm
                    pass

        data_updated = False
        if by is not Undefined:
            self._connection_size_by = by

            if by is None:
                self._encodings.delete('connection_size')

            else:
                self._encodings.set('connection_size', by)

                if not self._encodings.data[by].prepared:
                    component = self._encodings.data[by].component
                    try:
                        check_encoding_dtype(self._data[by])
                        if pd.api.types.is_categorical_dtype(self._data[by].dtype):
                            self._points[:, component] = self._data[by].cat.codes
                            self._connection_size_categories = dict(zip(self._data[by], self._data[by].cat.codes))
                        else:
                            self._points[:, component] = self._connection_size_norm(self._data[by].values)
                    except TypeError:
                        self._points[:, component] = self._connection_size_norm(np.asarray(by))

                    data_updated = True

                    # Make sure we don't prepare the data twice
                    self._encodings.data[by].prepared = True

            self.update_widget('connection_size_by', self.js_connection_size_by)

        if order is not Undefined:
            if order in [None, 'reverse']:
                self._connection_size_order = order
            elif self._connection_size_categories is not None:
                # Define order of the sizes instead of changing `points[:, component_idx]`
                self._connection_size_order = [self._connection_size_categories[cat] for cat in order]

        if map is not Undefined and map != 'auto':
            if type(map) == tuple:
                # Assuming `map` is a triple specifying a linear space
                self._connection_size_map = np.linspace(*map)
            else:
                self._connection_size_map = np.asarray(map)

        if (self._connection_size_map is None or map == 'auto') and self._connection_size_by is not None:
            # The best we can do is provide a linear size map
            if self._connection_size_categories is None:
                self._connection_size_map = np.linspace(1, 10, 19)
            else:
                self._connection_size_map = np.arange(1, len(self._connection_size_categories) + 1)

        self._connection_size_map = tolist(self._connection_size_map)

        # Update widget
        if self._connection_size_by is not None and self._connection_size_map is not None:
            self.update_widget(
                'connection_size',
                order_map(self._connection_size_map, self._connection_size_order)
            )
        else:
            self.update_widget('connection_size', self._connection_size)

        if data_updated and 'skip_widget_update' not in kwargs:
            self.update_widget('points', self.get_point_list())

        if self._connection_size_categories is not None:
            assert len(self._connection_size_categories) <= len(self._connection_size_map), 'More categories than connection sizes'

        if any_not([size, by, map, norm, order], Undefined):
            return self

        return dict(
            size = self._connection_size,
            by = self._connection_size_by,
            map = self._connection_size_map,
            norm = self._connection_size_norm,
            order = self._connection_size_order,
        )

    def background(self, color = Undefined, image = Undefined, **kwargs):
        if color is not Undefined:
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

        if image is not Undefined:
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

        if any_not([color, image], Undefined):
            return self

        return dict(
            color = self._background_color,
            image = self._background_image,
        )

    def camera(
        self,
        target = Undefined,
        distance = Undefined,
        rotation = Undefined,
        view = Undefined,
    ):
        if target is not Undefined:
            self._camera_target = target
            self.update_widget('camera_target', self._camera_target)

        if distance is not Undefined:
            try:
                self._camera_distance = float(distance)
                assert self._camera_distance > 0, 'Camera distance must be positive'
                self.update_widget('camera_distance', self._camera_distance)
            except ValueError:
                pass

        if rotation is not Undefined:
            try:
                self._camera_rotation = float(rotation)
                self.update_widget('camera_rotation', self._camera_rotation)
            except ValueError:
                pass

        if view is not Undefined:
            self._camera_view = view
            self.update_widget('camera_view', self._camera_view)

        if any_not([target, distance, rotation, view], Undefined):
            return self

        return dict(
            target = self._camera_target,
            distance = self._camera_distance,
            rotation = self._camera_rotation,
            view = self._camera_view,
        )

    def lasso(
        self,
        color = Undefined,
        initiator = Undefined,
        min_delay = Undefined,
        min_dist = Undefined,
    ):
        if color is not Undefined:
            try:
                self._lasso_color = to_rgba(color)
                self.update_widget('lasso_color', self._lasso_color)
            except:
                pass

        if initiator is not Undefined:
            try:
                self._lasso_initiator = bool(initiator)
                self.update_widget('lasso_initiator', self._lasso_initiator)
            except:
                pass

        if min_delay is not Undefined:
            try:
                self._lasso_min_delay = to_rgba(color)
                self.update_widget('lasso_min_delay', self._lasso_min_delay)
            except:
                pass

        if min_dist is not Undefined:
            try:
                self._lasso_min_dist = float(min_dist)
                self.update_widget('lasso_min_dist', self._lasso_min_dist)
            except:
                pass

        if any_not([color, initiator, min_delay, min_dist], Undefined):
            return self

        return dict(
            color = self._lasso_color,
            initiator = self._lasso_initiator,
            min_delay = self._lasso_min_delay,
            min_dist = self._lasso_min_dist,
        )

    def width(self, width = Undefined):
        if width is not Undefined:
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

    def height(self, height = Undefined):
        if height is not Undefined:
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

    def reticle(self, show = Undefined, color = Undefined):
        if show is not Undefined:
            try:
                self._reticle = bool(show)
                self.update_widget('reticle', self._reticle)
            except:
                pass

        if color is not Undefined:
            if color == 'auto':
                self._reticle_color = 'auto'
                self.update_widget('reticle_color', self.get_reticle_color())
            else:
                try:
                    self._reticle_color = to_rgba(color)
                    self.update_widget('reticle_color', self.get_reticle_color())
                except:
                    pass

        if any_not([show, color], Undefined):
            return self

        return dict(
            show = self._reticle,
            color = self._reticle_color,
        )

    def mouse(self, mode = Undefined):
        if mode is not Undefined:
            try:
                self._mouse_mode = mode
                self.update_widget('mouse_mode', mode)
            except:
                pass

            return self

        return self._mouse_mode

    def options(self, options = Undefined):
        if options is not Undefined:
            try:
                self._options = options
                self.update_widget('other_options', options)
            except:
                pass

            return self

        return self._options

    def pixels(self):
        if self._widget is not None:
            assert len(self._widget.view_pixels) > 0, 'Download pixels first'
            assert len(self._widget.view_shape) != 2, 'Download pixels first'

            self._pixels = np.asarray(self._widget.view_pixels).astype(np.uint8)
            self._pixels = self._pixels.reshape([*self._widget.view_shape, 4])

        return self._pixels

    @property
    def js_color_by(self):
        if self._color_by is not None:
            return component_idx_to_name(
                self._encodings.data[self._color_by].component
            )

        return None

    @property
    def js_opacity_by(self):
        if self._opacity_by == 'density':
            return 'density'

        elif self._opacity_by is not None:
            return component_idx_to_name(
                self._encodings.data[self._opacity_by].component
            )

        return None

    @property
    def js_size_by(self):
        if self._size_by is not None:
            return component_idx_to_name(
                self._encodings.data[self._size_by].component
            )

        return None

    @property
    def js_connection_color_by(self):
        if self._connection_color_by == 'segment':
            return 'segment'

        elif self._connection_color_by is not None:
            return component_idx_to_name(
                self._encodings.data[self._connection_color_by].component
            )

        return None

    @property
    def js_connection_opacity_by(self):
        if self._connection_opacity_by is not None:
            return component_idx_to_name(
                self._encodings.data[self._connection_opacity_by].component
            )

        return None

    @property
    def js_connection_size_by(self):
        if self._connection_size_by is not None:
            return component_idx_to_name(
                self._encodings.data[self._connection_size_by].component
            )

        return None

    @property
    def widget(self):
        if self._widget is not None:
            return self._widget

        self._widget = JupyterScatter(
            points=self.get_point_list(),
            selection=self._selection,
            width=self._width,
            height=self._height,
            background_color=self._background_color,
            background_image=self._background_image,
            lasso_color=self._lasso_color,
            lasso_initiator=self._lasso_initiator,
            lasso_min_delay=self._lasso_min_delay,
            lasso_min_dist=self._lasso_min_dist,
            color=order_map(self._color_map, self._color_order) if self._color_map else self._color,
            color_active=self._color_active,
            color_hover=self._color_hover,
            color_by=self.js_color_by,
            opacity=order_map(self._opacity_map, self._opacity_order) if self._opacity_map else self._opacity,
            opacity_by=self.js_opacity_by,
            size=order_map(self._size_map, self._size_order) if self._size_map else self._size,
            size_by=self.js_size_by,
            connect=bool(self._connect_by),
            connection_color=order_map(self._connection_color_map, self._connection_color_order) if self._connection_color_map else self._connection_color,
            connection_color_active=self._connection_color_active,
            connection_color_hover=self._connection_color_hover,
            connection_color_by=self.js_connection_color_by,
            connection_opacity=order_map(self._connection_opacity_map, self._connection_opacity_order) if self._connection_opacity_map else self._connection_opacity,
            connection_opacity_by=self.js_connection_opacity_by,
            connection_size=order_map(self._connection_size_map, self._connection_size_order) if self._connection_size_map else self._connection_size,
            connection_size_by=self.js_connection_size_by,
            reticle=self._reticle,
            reticle_color=self.get_reticle_color(),
            camera_target=self._camera_target,
            camera_distance=self._camera_distance,
            camera_rotation=self._camera_rotation,
            camera_view=self._camera_view,
            mouse_mode=self._mouse_mode,
            x_domain=self._x_domain,
            y_domain=self._y_domain,
            other_options=self._options
        )

        return self._widget

    def update_widget(self, prop, val):
        if self._widget is not None:
            setattr(self._widget, prop, val)

    def show(self):
        return self.widget.show()


def plot(x, y, data = None, **kwargs):
    return Scatter(x, y, data, **kwargs).show()
