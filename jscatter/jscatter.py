import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import typing as t

from matplotlib.colors import to_rgba

from .widget import JupyterScatter

from .color_maps import okabe_ito, glasbey_light, glasbey_dark

from .utils import any_not_none

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

def get_high_contrast_color(luminance):
    if luminance < 0.5:
        return (1, 1, 1, 1)

    return (0, 0, 0, 1)

def add_point_data_encoding(encodings, encoding):
    n = len(encodings)

    if encoding not in encodings:
        assert n < 2, 'Only two data-driven encodings are supported'
        # The first 2 components are the x and y coordinate
        encodings[encoding] = (2 + len(encodings), False)


class Scatter():
    def __init__(self, x, y, data = None, **kwargs):
        self._data = data

        try:
            self._n = len(self._data)
        except TypeError:
            self._n = np.asarray(x).size

        self._points = np.zeros((self._n, 5))

        self.x(x)
        self.y(y)

        # Default values
        self._widget = None
        self._pixels = None
        self._point_data_encodings = {}
        self._selection = np.asarray([])
        self._background_color = (1, 1, 1, 1)
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
        self._color_norm = matplotlib.colors.Normalize(0, 1, clip=True)
        self._color_order = None
        self._color_categories = None
        self._opacity = 0.66
        self._opacity_by = 'density'
        self._opacity_map = None
        self._opacity_norm = matplotlib.colors.Normalize(0, 1, clip=True)
        self._opacity_order = None
        self._opacity_categories = None
        self._size = 3
        self._size_by = None
        self._size_map = None
        self._size_norm = matplotlib.colors.Normalize(0, 1, clip=True)
        self._size_order = None
        self._size_categories = None
        self._connect_by = None
        self._connect_order = None
        self._connection_color = (0, 0, 0, 0.1)
        self._connection_color_active = (0, 0.55, 1, 1)
        self._connection_color_hover = (0, 0, 0, 0.66)
        self._connection_color_by = None
        self._connection_color_map = None
        self._connection_color_norm = matplotlib.colors.Normalize(0, 1, clip=True)
        self._connection_color_order = None
        self._connection_color_categories = None
        self._connection_opacity = 0.1
        self._connection_opacity_by = None
        self._connection_opacity_map = None
        self._connection_opacity_norm = matplotlib.colors.Normalize(0, 1, clip=True)
        self._connection_opacity_order = None
        self._connection_opacity_categories = None
        self._connection_size = 2
        self._connection_size_by = None
        self._connection_size_map = None
        self._connection_size_norm = matplotlib.colors.Normalize(0, 1, clip=True)
        self._connection_size_order = None
        self._connection_size_categories = None
        self._height = 240
        self._reticle = True
        self._reticle_color = (0, 0, 0, 0.1)
        self._camera_target = [0, 0]
        self._camera_distance = 1.0
        self._camera_rotation = 0.0
        self._camera_view = None
        self._mouse_mode = 'panZoom'
        self._sort_order = None
        self._options = {}

        #
        self.height(kwargs.get('height'))
        self.selection(
            kwargs.get('selection'),
        )
        self.color(
            kwargs.get('color'),
            kwargs.get('color_active'),
            kwargs.get('color_hover'),
            kwargs.get('color_by'),
            kwargs.get('color_map'),
            kwargs.get('color_norm'),
            kwargs.get('color_order'),
        )
        self.opacity(
            kwargs.get('opacity'),
            kwargs.get('opacity_by'),
            kwargs.get('opacity_map'),
            kwargs.get('opacity_norm'),
            kwargs.get('opacity_order'),
        )
        self.size(
            kwargs.get('size'),
            kwargs.get('size_by'),
            kwargs.get('size_map'),
            kwargs.get('size_norm'),
            kwargs.get('size_order'),
        )
        self.connect(kwargs.get('connect_by'), kwargs.get('connect_order'))
        self.connection_color(
            kwargs.get('connection_color'),
            kwargs.get('connection_color_active'),
            kwargs.get('connection_color_hover'),
            kwargs.get('connection_color_by'),
            kwargs.get('connection_color_map'),
            kwargs.get('connection_color_norm'),
            kwargs.get('connection_color_order'),
        )
        self.connection_opacity(
            kwargs.get('connection_opacity'),
            kwargs.get('connection_color_by'),
            kwargs.get('connection_color_map'),
            kwargs.get('connection_color_norm'),
            kwargs.get('connection_color_order'),
        )
        self.connection_size(
            kwargs.get('connection_size'),
            kwargs.get('connection_size_by'),
            kwargs.get('connection_size_map'),
            kwargs.get('connection_size_order'),
        )
        self.lasso(
            kwargs.get('lasso_color'),
            kwargs.get('lasso_initiator'),
            kwargs.get('lasso_min_delay'),
            kwargs.get('lasso_min_dist'),
        )
        self.reticle(kwargs.get('reticle'), kwargs.get('reticle_color'))
        self.mouse(kwargs.get('mouse_mode'))
        self.camera(
            kwargs.get('camera_target'),
            kwargs.get('camera_distance'),
            kwargs.get('camera_rotation'),
            kwargs.get('camera_view'),
        )
        self.options(kwargs.get('options'))

    def x(self, x = None):
        if x is not None:
            self._x = x

            try:
                self._points[:, 0] = self._data[x].values
            except TypeError:
                self._points[:, 0] = np.asarray(x)

            # Normalize x coordinate to [-1,1]
            self._x_min = np.min(self._points[:, 0])
            self._x_max = np.max(self._points[:, 0])
            self._x_extent = self._x_max - self._x_min
            self._points[:, 0] = (self._points[:, 0] - self._x_min) / self._x_extent * 2 - 1
            self._x_domain = [self._x_min, self._x_max]

            return self

        return self._x

    def y(self, y = None):
        if y is not None:
            self._y = y

            try:
                self._points[:, 1] = self._data[y].values
            except TypeError:
                self._points[:, 1] = np.asarray(y)

            # Normalize y coordinate to [-1,1]
            self._y_min = np.min(self._points[:, 1])
            self._y_max = np.max(self._points[:, 1])
            self._y_extent = self._y_max - self._y_min
            self._points[:, 1] = (self._points[:, 1] - self._y_min) / self._y_extent * 2 - 1
            self._y_domain = [self._y_min, self._y_max]

            return self

        return self._y

    def selection(self, selection = None):
        if selection is not None:
            try:
                self._selection = np.asarray(selection)
                self.update_widget('selection', self._selection.tolist())
            except:
                pass

            return self

        if self._widget is not None:
            return np.asarray(self._widget.selection)

        return self._selection

    def color(self, color = None, color_active = None, color_hover = None, by = None, map = None, norm = None, order = None):
        if color is not None:
            try:
                self._color = to_rgba(color)
                self.update_widget('color', self._color_map or self._color)
            except ValueError:
                pass

        if color_active is not None:
            try:
                self._color_active = to_rgba(color_active)
                self.update_widget('color_active', self._color_active)
            except ValueError:
                pass

        if color_hover is not None:
            try:
                self._color_hover = to_rgba(color_hover)
                self.update_widget('color_hover', self._color_hover)
            except ValueError:
                pass

        if norm is not None:
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
                    pass

        if by is not None:
            add_point_data_encoding(self._point_data_encodings, by)
            self._color_by = by;

            component_idx, is_component_prepared = self._point_data_encodings[by]
            if not is_component_prepared:
                try:
                    if self._data[by].dtype.name == 'category':
                        self._color_categories = dict(zip(self._data[by], self._data[by].cat.codes))
                        self._points[:, component_idx] = self._data[by].cat.codes
                    else:
                        self._points[:, component_idx] = self._color_norm(self._data[by].values)
                except TypeError:
                    self._points[:, component_idx] = self._color_norm(np.asarray(by))

                # Make sure we don't prepare the data twice
                self._point_data_encodings[by] = (component_idx, True)

            if order is not None and self._color_categories is not None:
                # Define order of the colors instead of changing `points[:, component_idx]`
                self._color_order = [self._color_categories[cat] for cat in order]

            if map is not None:
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

                    if self._color_order is not None:
                        # Reorder colors in case `self._color_order` is a list
                        try:
                            self._color_map = [map[self._color_order[i]] for i, _ in enumerate(self._color_map)]
                        except TypeError:
                            pass

            if self._color_map is None:
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

            # Reverse if needed
            self._color_map = self._color_map[::(1 + (-2 * (order == 'reverse')))]

            if self._color_categories is not None:
                assert len(self._color_categories) <= len(self._color_map), 'More categories than colors'

        if any_not_none([color, color_active, color_hover, by, map, norm, order]):
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

    def opacity(self, opacity = None, by = None, map = None, norm = None, order = None):
        if opacity is not None:
            try:
                self._opacity = float(opacity)
                assert self._opacity >= 0 and self._opacity <= 1, 'Opacity must be in [0,1]'
                self.update_widget('opacity', self._opacity_map or self._opacity)
            except ValueError:
                pass

        if norm is not None:
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
                    pass

        if by is not None:
            add_point_data_encoding(self._point_data_encodings, by)
            self._opacity_by = by;

            component_idx, is_component_prepared = self._point_data_encodings[by]
            if not is_component_prepared:
                try:
                    if self._data[by].dtype.name == 'category':
                        self._points[:, component_idx] = self._data[by].cat.codes
                        self._opacity_categories = dict(zip(self._data[by], self._data[by].cat.codes))
                    else:
                        self._points[:, component_idx] = self._opacity_norm(self._data[by].values)
                except TypeError:
                    self._points[:, component_idx] = self._opacity_norm(np.asarray(by))

                # Make sure we don't prepare the data twice
                self._point_data_encodings[by] = (component_idx, True)

            if order is not None:
                if order == 'reverse':
                    self._opacity_order = order
                elif self._opacity_categories is not None:
                    # Define order of the opacities instead of changing `points[:, component_idx]`
                    self._opacity_order = [self._opacity_categories[cat] for cat in order]

            if map is not None:
                if type(map) == tuple:
                    # Assuming `map` is a triple specifying a linear space
                    start, end, num = map
                    self._opacity_map = np.linspace(start, end, num)
                else:
                    self._opacity_map = np.asarray(map)

                if self._opacity_categories is not None and self._opacity_order is not None:
                    try:
                        self._opacity_map = np.asarray([
                            self._opacity_map[self._opacity_order[i]] for i, _ in enumerate(self._opacity_map)
                        ])
                    except TypeError:
                        pass

            if self._opacity_map is None:
                # The best we can do is provide a linear opacity map
                if self._opacity_categories is not None:
                    self._opacity_map = np.linspace(1/len(self._opacity_categories), 1, len(self._opacity_categories))
                else:
                    self._opacity_map = np.linspace(1/256, 1, 256)

            # Reverse if needed
            self._opacity_map = self._opacity_map[::(1 + (-2 * (self._opacity_order == 'reverse')))]
            self._opacity_map = self._opacity_map.tolist()

            if self._opacity_categories is not None:
                assert len(self._opacity_categories) <= len(self._opacity_map), 'More categories than opacities'

        if any_not_none([opacity, by, map, norm, order]):
            return self

        return dict(
            opacity = self._opacity,
            by = self._opacity_by,
            map = self._opacity_map,
            norm = self._opacity_norm,
            order = self._opacity_order,
        )

    def size(self, size = None, by = None, map = None, norm = None, order = None):
        if size is not None:
            try:
                self._size = int(size)
                assert self._size > 0, 'Size must be a positive integer'
                self.update_widget('size', self._size_map or self._size)
            except ValueError:
                pass

        if norm is not None:
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
                    pass

        if by is not None:
            add_point_data_encoding(self._point_data_encodings, by)
            self._size_by = by

            component_idx, is_component_prepared = self._point_data_encodings[by]
            if not is_component_prepared:
                try:
                    if self._data[by].dtype.name == 'category':
                        self._points[:, component_idx] = self._data[by].cat.codes
                        self._size_categories = dict(zip(self._data[by], self._data[by].cat.codes))
                    else:
                        self._points[:, component_idx] = self._size_norm(self._data[by].values)
                except TypeError:
                    self._points[:, component_idx] = self._size_norm(np.asarray(by))

                # Make sure we don't prepare the data twice
                self._point_data_encodings[by] = (component_idx, True)

            if order is not None:
                if order == 'reverse':
                    self._size_order = order
                elif self._size_categories is not None:
                    # Define order of the sizes instead of changing `points[:, component_idx]`
                    self._size_order = [self._size_categories[cat] for cat in self._size_order]

            if map is not None:
                if type(map) == tuple:
                    # Assuming `map` is a triple specifying a linear space
                    start, end, num = map
                    self._size_map = np.linspace(start, end, num)
                else:
                    self._size_map = np.asarray(map)

                if self._size_categories is not None and self._size_order is not None:
                    try:
                        self._size_map = np.asarray([
                            self._size_map[self._size_order[i]] for i, _ in enumerate(self._size_map)
                        ])
                    except TypeError:
                        pass

            if self._size_map is None:
                # The best we can do is provide a linear size map
                if self._size_categories is None:
                    self._size_map = np.linspace(1, 10, 19)
                else:
                    self._size_map = np.arange(1, len(self._size_categories) + 1)

            # Reverse if needed
            self._size_map = self._size_map[::(1 + (-2 * (self._size_order == 'reverse')))]
            self._size_map = self._size_map.tolist()

            if self._size_categories is not None:
                assert len(self._size_categories) <= len(self._size_map), 'More categories than sizes'

        if any_not_none([size, by, map, norm, order]):
            return self

        return dict(
            size = self._size,
            by = self._size_by,
            map = self._size_map,
            norm = self._size_norm,
            order = self._size_order,
        )

    def connect(self, by = None, order = None):
        if by is not None:
            self._connect_by = by
            categories = None

            try:
                if self._data[by].dtype.name == 'category':
                    self._points[:, 4] = self._data[by].cat.codes
                    categories = dict(zip(self._data[by], self._points[:, 4]))
                else:
                    raise TypeError('connect by only works with categorical data')
            except TypeError:
                tmp = pd.Series(by, dtype='category')
                self._points[:, 4] = tmp.cat.codes
                categories = dict(zip(tmp, tmp.cat.codes))

            assert categories is not None, 'connect by data is broken. everybody: ruuuun!'

        if order is not None:
            # Since regl-scatterplot doesn't support `order` we have to sort the data now
            try:
                # Sort data
                sorting = self._data.sort_values([by, order]).index.values
                self._data = self._data[sorting]
                self._sort_order = sorting_to_dict(sorting)
            except TypeError:
                raise TypeError('connect order only works with Pandas data for now')

        if any_not_none([by, order]):
            return self

        return dict(
            by = self._connect_by,
            order = self._connect_order,
        )

    def connection_color(self, color = None, color_active = None, color_hover = None, by = None, map = None, norm = None, order = None):
        if color is not None:
            try:
                self._connection_color = to_rgba(color)
                self.update_widget('connection_color', self._connection_color_map or self._connection_color)
            except ValueError:
                pass

        if color_active is not None:
            try:
                self._connection_color_active = to_rgba(color_active)
                self.update_widget('connection_color_active', self._connection_color_active)
            except ValueError:
                pass

        if color_hover is not None:
            try:
                self._connection_color_hover = to_rgba(color_hover)
                self.update_widget('connection_color_hover', self._connection_color_hover)
            except ValueError:
                pass

        if norm is not None:
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
                    pass

        if by is not None:
            add_point_data_encoding(self._point_data_encodings, by)
            self._connection_color_by = by

            component_idx, is_component_prepared = self._point_data_encodings[by]
            if not is_component_prepared:
                try:
                    if self._data[by].dtype.name == 'category':
                        self._connection_color_categories = dict(zip(self._data[by], self._data[by].cat.codes))
                        self._points[:, component_idx] = self._data[by].cat.codes
                    else:
                        self._points[:, component_idx] = self._connection_color_norm(self._data[by].values)
                except TypeError:
                    self._points[:, component_idx] = self._connection_color_norm(np.asarray(by))

                # Make sure we don't prepare the data twice
                self._point_data_encodings[by] = (component_idx, True)

            if order is not None:
                if order == 'reverse':
                    self._connection_color_order = order
                elif self._connection_color_categories is not None:
                    # Define order of the colors instead of changing `points[:, component_idx]`
                    self._connection_color_order = [self._connection_color_categories[cat] for cat in order]

            if map is not None:
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

                    if self._connection_color_order is not None:
                        try:
                            self._connection_color_map = [self._connection_color_map[self._connection_color_order[i]] for i, _ in enumerate(self._connection_color_map)]
                        except TypeError:
                            pass
            if self._connection_color_map is None:
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

            # Reverse if needed
            self._connection_color_map = self._connection_color_map[::(1 + (-2 * (self._connection_color_order == 'reverse')))]

            if self._connection_color_categories is not None:
                assert len(self._connection_color_categories) <= len(self._connection_color_map), 'More categories than connection colors'

        if any_not_none([color, by, map, norm, order]):
            return self

        return dict(
            color = self._connection_color,
            by = self._connection_color_by,
            map = self._connection_color_map,
            norm = self._connection_color_norm,
            order = self._connection_color_order,
        )

    def connection_opacity(self, opacity = None, by = None, map = None, norm = None, order = None):
        if opacity is not None:
            try:
                self._connection_opacity = float(opacity)
                assert self._connection_opacity >= 0 and self._connection_opacity <= 1, 'Connection opacity must be in [0,1]'
                self.update_widget('connection_opacity', self._connection_opacity_map or self._connection_opacity)
            except ValueError:
                pass

        if norm is not None:
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
                    pass

        if by is not None:
            add_point_data_encoding(self._point_data_encodings, by)
            self._connection_opacity_by = by

            component_idx, is_component_prepared = self._point_data_encodings[by]
            if not is_component_prepared:
                try:
                    if self._data[by].dtype.name == 'category':
                        self._points[:, component_idx] = self._data[by].cat.codes
                        self._connection_opacity_categories = dict(zip(self._data[by], self._data[by].cat.codes))
                    else:
                        self._points[:, component_idx] = self._connection_opacity_norm(self._data[by].values)
                except TypeError:
                    self._points[:, component_idx] = self._connection_opacity_norm(np.asarray(by))

                # Make sure we don't prepare the data twice
                self._point_data_encodings[by] = (component_idx, True)

            if order is not None:
                if order == 'reverse':
                    self._connection_opacity_order = order
                elif self._connection_opacity_categories is not None:
                    # Define order of the opacities instead of changing `points[:, component_idx]`
                    self._connection_opacity_order = [self._connection_opacity_categories[cat] for cat in order]

            if map is not None:
                if type(map) == tuple:
                    # Assuming `map` is a triple specifying a linear space
                    start, end, num = map
                    self._connection_opacity_map = np.linspace(start, end, num)
                else:
                    self._connection_opacity_map = np.asarray(map)

                if self._connection_opacity_categories is not None and self._connection_opacity_order is not None:
                    try:
                        self._connection_opacity_map = np.asarray([
                            self._connection_opacity_map[self._connection_opacity_order[i]]
                            for i, _ in enumerate(self._connection_opacity_map)
                        ])
                    except TypeError:
                        pass

            else:
                # The best we can do is provide a linear opacity map
                if self._connection_opacity_categories is not None:
                    self._connection_opacity_map = np.linspace(
                        1 / len(self._connection_opacity_categories),
                        1,
                        len(self._connection_opacity_categories)
                    )
                else:
                    self._connection_opacity_map = np.linspace(1/256, 1, 256)

            # Reverse if needed
            self._connection_opacity_map = self._connection_opacity_map[::(1 + (-2 * (self._connection_opacity_order == 'reverse')))]
            self._connection_opacity_map = self._connection_opacity_map.tolist()

            if self._connection_opacity_categories is not None:
                assert len(self._connection_opacity_categories) <= len(self._connection_opacity_map), 'More categories than connection opacities'

        if any_not_none([opacity, by, map, norm, order]):
            return self

        return dict(
            opacity = self._connection_opacity,
            by = self._connection_opacity_by,
            map = self._connection_opacity_map,
            norm = self._connection_opacity_norm,
            order = self._connection_opacity_order,
        )

    def connection_size(self, size = None, by = None, map = None, norm = None, order = None):
        if size is not None:
            try:
                self._connection_size = int(size)
                assert self._connection_size > 0, 'Connection size must be a positive integer'
                self.update_widget('connection_size', self._connection_size_map or self._connection_size)
            except ValueError:
                pass

        if norm is not None:
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
                    pass

        if by is not None:
            add_point_data_encoding(self._point_data_encodings, by)
            self._connection_size_by = by

            component_idx, is_component_prepared = self._point_data_encodings[by]
            if not is_component_prepared:
                try:
                    if self._data[by].dtype.name == 'category':
                        self._points[:, component_idx] = self._data[by].cat.codes
                        self._connection_size_categories = dict(zip(self._data[by], self._data[by].cat.codes))
                    else:
                        self._points[:, component_idx] = self._connection_size_norm(self._data[by].values)
                except TypeError:
                    self._points[:, component_idx] = self._connection_size_norm(np.asarray(by))

                # Make sure we don't prepare the data twice
                self._point_data_encodings[by] = (component_idx, True)

            if order is not None:
                if order == 'reverse':
                    self._connection_size_order = order
                elif self._connection_size_categories is not None:
                    # Define order of the sizes instead of changing `points[:, component_idx]`
                    self._connection_size_order = [self._connection_size_categories[cat] for cat in order]

            if map is not None:
                if type(map) == tuple:
                    # Assuming `map` is a triple specifying a linear space
                    start, end, num = map
                    self._connection_size_map = np.linspace(start, end, num)
                else:
                    self._connection_size_map = np.asarray(map)

                if self._connection_size_categories is not None and self._connection_size_order is not None:
                    try:
                        self._connection_size_map = np.asarray([self._connection_size_map[self._connection_size_order[i]] for i, _ in enumerate(self._connection_size_map)])
                    except TypeError:
                        pass

            if self._connection_size_map is None:
                # The best we can do is provide a linear size map
                if self._connection_size_categories is None:
                    self._connection_size_map = np.linspace(1, 10, 19)
                else:
                    self._connection_size_map = np.arange(1, len(self._connection_size_categories) + 1)

            # Reverse if needed
            self._connection_size_map = self._connection_size_map[::(1 + (-2 * (self._connection_size_order == 'reverse')))]
            self._connection_size_map = self._connection_size_map.tolist()

            if self._connection_size_categories is not None:
                assert len(self._connection_size_categories) <= len(self._connection_size_map), 'More categories than connection sizes'

        if any_not_none([size, by, map, norm, order]):
            return self

        return dict(
            size = self._connection_size,
            by = self._connection_size_by,
            map = self._connection_size_map,
            norm = self._connection_size_norm,
            order = self._connection_size_order,
        )

    def background(self, color = None, image = None, **kwargs):
        if color is not None:
            try:
                self._background_color = to_rgba(color)
                self.update_widget('background_color', self._background_color)
            except:
                pass

            self._background_color_luminance = math.sqrt(
                0.299 * self._background_color[0] ** 2
                + 0.587 * self._background_color[1] ** 2
                + 0.114 * self._background_color[2] ** 2
            )

        if image is not None:
            try:
                im = plt.imshow(image, **kwargs)

                x = im.make_image()
                h, w, d = x.as_rgba_str()
                self._background_image = np.fromstring(d, dtype=np.uint8).reshape(h, w, 4)
                self.update_widget('background_image', self._background_image)
            except:
                pass

        if any_not_none([color, image]):
            return self

        return dict(
            color = self._background_color,
            image = self._background_image,
        )

    def camera(self, target = None, distance = None, rotation = None, view = None):
        if target is not None:
            self._camera_target = target
            self.update_widget('camera_target', self._camera_target)

        if distance is not None:
            try:
                self._camera_distance = float(distance)
                assert self._camera_distance > 0, 'Camera distance must be positive'
                self.update_widget('camera_distance', self._camera_distance)
            except ValueError:
                pass

        if rotation is not None:
            try:
                self._camera_rotation = float(rotation)
                self.update_widget('camera_rotation', self._camera_rotation)
            except ValueError:
                pass

        if view is not None:
            self._camera_view = view
            self.update_widget('camera_view', self._camera_view)

        if any_not_none([target, distance, rotation, view]):
            return self

        return dict(
            target = self._camera_target,
            distance = self._camera_distance,
            rotation = self._camera_rotation,
            view = self._camera_view,
        )

    def lasso(self, color = None, initiator = None, min_delay = None, min_dist = None):
        if color is not None:
            try:
                self._lasso_color = to_rgba(color)
                self.update_widget('lasso_color', self._lasso_color)
            except:
                pass

        if initiator is not None:
            try:
                self._lasso_initiator = bool(initiator)
                self.update_widget('lasso_initiator', self._lasso_initiator)
            except:
                pass

        if min_delay is not None:
            try:
                self._lasso_min_delay = to_rgba(color)
                self.update_widget('lasso_min_delay', self._lasso_min_delay)
            except:
                pass

        if min_dist is not None:
            try:
                self._lasso_min_dist = float(min_dist)
                self.update_widget('lasso_min_dist', self._lasso_min_dist)
            except:
                pass

        if any_not_none([color, initiator, min_delay, min_dist]):
            return self

        return dict(
            color = self._lasso_color,
            initiator = self._lasso_initiator,
            min_delay = self._lasso_min_delay,
            min_dist = self._lasso_min_dist,
        )

    def height(self, height = None):
        """Get or set height

        Parameters
        ----------
        height : int
            Height of the scatter plot

        Returns
        -------
        scatter plot height or widget
        """
        if height is not None:
            try:
                self._height = int(height)
                self.update_widget('height', self._height)
            except:
                pass

            return self

        return self._height

    def reticle(self, show = None, color = None):
        if show is not None:
            try:
                self._reticle = bool(show)
                self.update_widget('reticle', self._reticle)
            except:
                pass

        if color is not None:
            try:
                self._reticle_color = to_rgba(color)
                self.update_widget('reticle_color', self._reticle_color)
            except:
                pass

        if any_not_none([show, color]):
            return self

        return dict(
            show = self._reticle,
            color = self._reticle_color,
        )

    def mouse(self, mode = None):
        if mode is not None:
            try:
                self._mouse_mode = mode
                self.update_widget('mouse_mode', mode)
            except:
                pass

            return self

        return self._mouse_mode

    def options(self, options = None):
        if options is not None:
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
    def widget(self):
        if self._widget is not None:
            return self._widget

        js_color_by = None
        if self._color_by is not None:
            js_color_by = component_idx_to_name(
                self._point_data_encodings[self._color_by][0]
            )

        js_opacity_by = None
        if self._opacity_by == 'density':
            js_opacity_by = 'density'
        elif self._opacity_by is not None:
            js_opacity_by = component_idx_to_name(
                self._point_data_encodings[self._opacity_by][0]
            )

        js_size_by = None
        if self._size_by is not None:
            js_size_by = component_idx_to_name(
                self._point_data_encodings[self._size_by][0]
            )

        js_connection_color_by = None
        if self._connection_color_by is not None:
            js_connection_color_by = component_idx_to_name(
                self._point_data_encodings[self._connection_color_by][0]
            )

        js_connection_opacity_by = None
        if self._connection_opacity_by is not None:
            js_connection_opacity_by = component_idx_to_name(
                self._point_data_encodings[self._connection_opacity_by][0]
            )

        js_connection_size_by = None
        if self._connection_size_by is not None:
            js_connection_size_by = component_idx_to_name(
                self._point_data_encodings[self._connection_size_by][0]
            )

        self._widget = JupyterScatter(
            points=self._points.tolist(),
            selection=self._selection.tolist(),
            height=self._height,
            background_color=self._background_color,
            background_image=self._background_image,
            lasso_color=self._lasso_color,
            lasso_initiator=self._lasso_initiator,
            lasso_min_delay=self._lasso_min_delay,
            lasso_min_dist=self._lasso_min_dist,
            color=self._color_map or self._color,
            color_active=self._color_active,
            color_hover=self._color_hover,
            color_by=js_color_by,
            opacity=self._opacity_map or self._opacity,
            opacity_by=js_opacity_by,
            size=self._size_map or self._size,
            size_by=js_size_by,
            connect=bool(self._connect_by),
            connection_color=self._connection_color_map or self._connection_color,
            connection_color_active=self._connection_color_active,
            connection_color_hover=self._connection_color_hover,
            connection_color_by=js_connection_color_by,
            connection_opacity=self._connection_opacity_map or self._connection_opacity,
            connection_opacity_by=js_connection_opacity_by,
            connection_size=self._connection_size_map or self._connection_size,
            connection_size_by=js_connection_size_by,
            reticle=self._reticle,
            reticle_color=self._reticle_color,
            camera_target=self._camera_target,
            camera_distance=self._camera_distance,
            camera_rotation=self._camera_rotation,
            camera_view=self._camera_view,
            mouse_mode=self._mouse_mode,
            x_domain=self._x_domain,
            y_domain=self._y_domain,
            other_options=self._options,
            sort_order=self._sort_order
        )

        return self._widget

    def update_widget(self, prop, val):
        if self._widget is not None:
            setattr(self._widget, prop, val)

    def show(self):
        return self.widget.show()

def plot(x, y, data = None, **kwargs):
    return Scatter(x, y, data, **kwargs).show()
