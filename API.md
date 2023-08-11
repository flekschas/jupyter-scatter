# API Docs

- [Scatter](#scatter)
  - [Methods](#methods)
    - [x()](#scatter.x), [y()](#scatter.x), [xy()](#scatter.xy), and [data()](#scatter.data)
    - [selection()](#scatter.selection), [filter()](#scatter.filter)
    - [color()](#scatter.color), [opacity()](#scatter.opacity), and [size()](#scatter.size)
    - [connect()](#scatter.connect), [connection_color()](#scatter.connection_color), [connection_opacity()](#scatter.connection_opacity), and [connection_size()](#scatter.connection_size)
    - [axes()](#scatter.axes) [legend()](#scatter.legend),  and [tooltip()](#scatter.tooltip)
    - [zoom()](#scatter.zoom) and [camera()](#scatter.camera)
    - [lasso()](#scatter.lasso), [reticle()](#scatter.reticle), and [mouse()](#scatter.mouse),
    - [background()](#scatter.background) and [options()](scatter.options)
  - [Properties](#properties)
  - [Widget](#widget)
- [Plotting](#plotting)
- [Composing \& Linking](#composing--linking)
- [Color Maps](#color-maps)

# Scatter

<h3><a name="Scatter" href="#Scatter">#</a> <b>Scatter</b>(<i>x</i>, <i>y</i>, <i>data = None</i>, <i>**kwargs</i>)</h3>

**Arguments:**

- `x` is either an array-like list of coordinates or a string referencing a column in `data`.
- `y` is either an array-like list of coordinates or a string referencing a column in `data`.
- `data` is a Pandas DataFrame. [optional]
- `kwargs` is a dictionary of additional [properties](#properties). [optional]

**Returns:** a new scatter instance.

**Examples:**

```python
from jscatter import Scatter
scatter = Scatter(x='speed', y='weight', data=cars)
scatter.show()
```

## Methods

<h3><a name="scatter.x" href="#scatter.x">#</a> scatter.<b>x</b>(<i>x=Undefined</i>, <i>scale=Undefined</i>, <i>**kwargs</i>)</h3>

Get or set the x coordinate.

**Arguments:**

- `x` is either an array-like list of coordinates or a string referencing a column in `data`.
- `scale` is either a string (`linear`, `log`, `pow`), a tuple defining the value range that's map to the extent of the scatter plot, or an instance of `matplotlib.colors.LogNorm` or `matplotlib.colors.PowerNorm`.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the x coordinate when x is `Undefined` or `self`.

**Examples:**

```python
scatter.x('price') # Triggers and animated transition of the x coordinates
```

<h3><a name="scatter.y" href="#scatter.y">#</a> scatter.<b>y</b>(<i>y=Undefined</i>, <i>scale=Undefined</i>, <i>**kwargs</i>)</h3>

Get or set the y coordinate.

**Arguments:**

- `y` is either an array-like list of coordinates or a string referencing a column in `data`.
- `scale` is either a string (`linear`, `log`, `pow`), a tuple defining the value range that's map to the extent of the scatter plot, or an instance of `matplotlib.colors.LogNorm` or `matplotlib.colors.PowerNorm`.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the y coordinate when y is `Undefined` or `self`.

**Examples:**

```python
scatter.y('price') # Triggers and animated transition of the y coordinates
```

<h3><a name="scatter.xy" href="#scatter.xy">#</a> scatter.<b>xy</b>(<i>x=Undefined</i>, <i>y=Undefined</i>, <i>x_scale=Undefined</i>, <i>y_scale=Undefined</i>, <i>**kwargs</i>)</h3>

Get or set the x and y coordinate. This is just a convenience function to animate a change in the x and y coordinate at the same time.

**Arguments:**

- `x` is either an array-like list of coordinates or a string referencing a column in `data`.
- `y` is either an array-like list of coordinates or a string referencing a column in `data`.
- `x_scale` is either a string (`linear`, `log`, `pow`), a tuple defining the value range that's map to the extent of the scatter plot, or an instance of `matplotlib.colors.LogNorm` or `matplotlib.colors.PowerNorm`.
- `y_scale` is either a string (`linear`, `log`, `pow`), a tuple defining the value range that's map to the extent of the scatter plot, or an instance of `matplotlib.colors.LogNorm` or `matplotlib.colors.PowerNorm`.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the x and y coordinate when x and y are `Undefined` or `self`.

**Examples:**

```python
scatter.xy('size', 'speed') # Mirror plot along the diagonal
```

<h3><a name="scatter.data" href="#scatter.data">#</a> scatter.<b>data</b>(<i>data=Undefined</i>, <i>use_index=Undefined</i>, <i>**kwargs</i>)</h3>

Get or set the referenced Pandas DataFrame. This is just a convenience function to animate a change in the x and y coordinate at the same time.

**Arguments:**

- `data` is a Pandas DataFrame.
- `use_index` is a Boolean value indicating if the data frame's index should be used for referencing point by the `selection()` and `filter()` methods instead of the row index.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the `data` and `use_index` if no argument was passed to the method or `self`.

**Examples:**

```python
scatter.data(df)
```

<h3><a name="scatter.selection" href="#scatter.selection">#</a> scatter.<b>selection</b>(<i>point_idxs=Undefined</i>)</h3>

Get or set the selected points.

**Arguments:**

- `point_idxs` is either an array-like list of point indices.

**Returns:** either the currently selected point indices when `point_idxs` is `Undefined` or `self`.

**Examples:**

```python
# Select all points corresponding to cars with a speed of less than 50
scatter.selection(cars.query('speed < 50').index)

# To unset the selection
scatter.selection(None) # or scatter.selection([])

# Retrieve the point indices of the currently selected points
scatter.selection()
# => array([0, 42, 1337], dtype=uint32)
```

<h3><a name="scatter.filter" href="#scatter.filter">#</a> scatter.<b>filter</b>(<i>point_idxs=Undefined</i>)</h3>

Get or set the filtered points. When filtering down to a set of points, all other points will be hidden from the view.

**Arguments:**

- `point_idxs` is a list or an array-like object of point indices or `None`.

**Returns:** either the currently filtered point indices when `point_idxs` is `Undefined` or `self`.

**Examples:**

```python
scatter.filter(cars.query('speed < 50').index)
scatter.filter(None) # To unset filter
```

<h3><a name="scatter.color" href="#scatter.color">#</a> scatter.<b>color</b>(<i>default=Undefined</i>, <i>selected=Undefined</i>, <i>hover=Undefined</i>, <i>by=Undefined</i>, <i>map=Undefined</i>, <i>norm=Undefined</i>, <i>order=Undefined</i>, <i>labeling=Undefined</i>, <i>**kwargs</i>)</h3>

Get or set the point color.

**Arguments:**

- `default` is a valid matplotlib color.
- `selected` is a valid matplotlib color.
- `hover` is a valid matplotlib color.
- `by` is either an array-like list of values or a string referencing a column in `data`.
- `map` is either a string referencing a matplotlib color map, a matplotlib color map object, a list of matplotlib-compatible colors, a dictionary of category-color pairs, or `auto` (to let jscatter choose a default color map).
- `norm` is either a tuple defining a value range that's map to `[0, 1]` with `matplotlib.colors.Normalize` or a [matplotlib normalizer](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.colors.Normalize.html).
- `order` is either a list of values (for categorical coloring) or `reverse` to reverse a color map.
- `labeling` is either a tuple of three strings specyfing a label for the minimum value, maximum value, and variable that the color encodes or a dictionary of the form `{'minValue': 'label', 'maxValue': 'label', 'variable': 'label'}`. The specified labels are only used for continuous color encoding and are displayed together with the legend.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the x and y coordinate when x and y are `Undefined` or `self`.

**Examples:**

```python
# Assuming `country` is of type `category` with less than nine categories, then
# the default color map will be Okabe Ito. Otherwise is it Glasbey. When the
# data type is not `category` then `viridis` is the default color map.
scatter.color(by='country')

# You can of course override the color map as follows.
scatter.color(
  by='country',
  map=dict(
    usa='red',
    europe='green',
    asia='blue'
  ),
)

# Assuming `gpd` is a continue float/int, we can also reference Matplotlib colormaps by their name
scatter.color(by='gpd', map='viridis')
```

<h3><a name="scatter.opacity" href="#scatter.opacity">#</a> scatter.<b>opacity</b>(<i>default=Undefined</i>, <i>unselected=Undefined</i>, <i>by=Undefined</i>, <i>map=Undefined</i>, <i>norm=Undefined</i>, <i>order=Undefined</i>, <i>labeling=Undefined</i>, <i>**kwargs</i>)</h3>

Get or set the point opacity.

**Arguments:**

- `default` is a valid matplotlib color.
- `unselected` is the factor by which the opacity of unselected points is scaled. It must be in [0, 1] and is only applied if one or more points are selected.
- `by` is either an array-like list of values, a string referencing a column in `data`, or `density`
- `map` is either a triple specifying an `np.linspace(*map)`, a list of opacities, a dictionary of category-opacity pairs, or `auto` (to let jscatter choose a default opacity map).
- `norm` is either a tuple defining a value range that's map to `[0, 1]` with `matplotlib.colors.Normalize` or a [matplotlib normalizer](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.colors.Normalize.html).
- `order` is either a list of values (for categorical opacity encoding) or `reverse` to reverse the opacity map.
- `labeling` is either a tuple of three strings specyfing a label for the minimum value, maximum value, and variable that the opacity encodes or a dictionary of the form `{'minValue': 'label', 'maxValue': 'label', 'variable': 'label'}`. The specified labels are only used for continuous opacity encoding and are displayed together with the legend.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the x and y coordinate when x and y are `Undefined` or `self`.

**Examples:**

```python
# Data-driven opacity encoding
scatter.opacity(by='price', map=(1, 0.25, 10))

# View-driven opacity encoding: the opacity is determined dynamically depending
# on the number and size of points in the view.
scatter.opacity(by='density')
```

<h3><a name="scatter.size" href="#scatter.size">#</a> scatter.<b>size</b>(<i>default=Undefined</i>, <i>by=Undefined</i>, <i>map=Undefined</i>, <i>norm=Undefined</i>, <i>order=Undefined</i>, <i>labeling=Undefined</i>, <i>**kwargs</i>)</h3>

Get or set the point size.

**Arguments:**

- `default` is a valid matplotlib color.
- `by` is either an array-like list of values or a string referencing a column in `data`.
- `map` is either a triple specifying an `np.linspace(*map)`, a list of sizes, a dictionary of category-size pairs, or `auto` (to let jscatter choose a default size map).
- `norm` is either a tuple defining a value range that's map to `[0, 1]` with `matplotlib.colors.Normalize` or a [matplotlib normalizer](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.colors.Normalize.html).
- `order` is either a list of values (for categorical size encoding) or `reverse` to reverse the size map.
- `labeling` is either a tuple of three strings specyfing a label for the minimum value, maximum value, and variable that the size encodes or a dictionary of the form `{'minValue': 'label', 'maxValue': 'label', 'variable': 'label'}`. The specified labels are only used for continuous size encoding and are displayed together with the legend.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the x and y coordinate when x and y are `Undefined` or `self`.

**Examples:**

```python
scatter.size(by='price', map=(1, 0.25, 10))
```

<h3><a name="scatter.connect" href="#scatter.connect">#</a> scatter.<b>connect</b>(<i>by=Undefined</i>, <i>order=Undefined</i>, <i>**kwargs</i>)</h3>

Get or set the point connection.

**Description:** The `by` argument defines which points are part of a line segment. Points with the same value are considered to be part of a line segment. By default, points are connected in the order in which they appear the dataframe. You can customize that ordering via `order`.

**Arguments:**

- `by` is either an array-like list of integers or a string referencing a column in the dataframe.
- `order` is either an array-like list of integers or a string referencing a column in the dataframe.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the connect properties when `by` and `order` are `Undefined` or `self`.

**Examples:**

Dataframe:

|   | x    | y    | group | order |
|---|------|------|-------|-------|
| 0 | 0.13 | 0.27 | A     | 2     |
| 1 | 0.87 | 0.93 | A     | 1     |
| 2 | 0.10 | 0.25 | B     | 2     |
| 3 | 0.03 | 0.90 | A     | 3     |
| 4 | 0.19 | 0.78 | B     | 1     |

```python
# The following call will result in two lines, connecting the points:
# - 0, 1, and 3
# - 2 and 4
scatter.connect(by='group')
# Note that the points will be connected by a line in the order in which they
# appear in the dataframe.

# To customize the order use the `order` column:
scatter.connect(by='group', order='order')
# This results in the following two lines:
# - [1]--[0]--[3]
# - [4]--[2]
```


<h3><a name="scatter.connection_color" href="#scatter.connection_color">#</a> scatter.<b>connection_color</b>(<i>default=Undefined</i>, <i>selected=Undefined</i>, <i>hover=Undefined</i>, <i>by=Undefined</i>, <i>map=Undefined</i>, <i>norm=Undefined</i>, <i>order=Undefined</i>, <i>labeling=Undefined</i>, <i>**kwargs</i>)</h3>

Get or set the point connection color. This function behaves identical to [scatter.color()][scatter.color].


<h3><a name="scatter.connection_opacity" href="#scatter.connection_opacity">#</a> scatter.<b>connection_opacity</b>(<i>default=Undefined</i>, <i>by=Undefined</i>, <i>map=Undefined</i>, <i>norm=Undefined</i>, <i>order=Undefined</i>, <i>labeling=Undefined</i>, <i>**kwargs</i>)</h3>

Get or set the point connection opacity. This function behaves identical to [scatter.opacity()][scatter.opacity].


<h3><a name="scatter.connection_size" href="#scatter.connection_size">#</a> scatter.<b>connection_size</b>(<i>default=Undefined</i>, <i>by=Undefined</i>, <i>map=Undefined</i>, <i>norm=Undefined</i>, <i>order=Undefined</i>, <i>labeling=Undefined</i>, <i>**kwargs</i>)</h3>

Get or set the point connection size. This function behaves identical to [scatter.size()][scatter.size].


<h3><a name="scatter.axes" href="#scatter.axes">#</a> scatter.<b>axes</b>(<i>axes=Undefined</i>, <i>grid=Undefined</i>, <i>labels=Undefined</i>)</h3>

Get or set the x and y axes.

**Arguments:**
- `axes` is a Boolean value to specify if the x and y axes should be shown or not.
- `grid` is a Boolean value to specify if an axes-based grid should be shown or not.
- `labels` is a Boolean value, a list of strings, or a dictionary with two keys (x and y) that specify the axes labels. When set to `True` the labels are the x and y column name of `data`.

**Returns:** either the axes properties when all arguments are `Undefined` or `self`.

**Example:**

```python
scatter = Scatter(data=df, x='speed', y='weight')
scatter.axes(axes=True, labels=['Speed (km/h)', 'Weight (tons)'])
```

<h3><a name="scatter.legend" href="#scatter.legend">#</a> scatter.<b>legend</b>(<i>legend=Undefined</i>, <i>position=Undefined</i>, <i>size=Undefined</i>)</h3>

Set or get the legend settings.

**Arguments:**
- `legend` is a Boolean specifying if the legend should be shown or not.
- `position` is a string specifying the legend position. It must be one of `top`, `left`, `right`, `bottom`, `top-left`, `top-right`, `bottom-left`, `bottom-right`, or `center`.
- `size` is a string specifying the size of the legend. It must be one of `small`, `medium`, or `large`.

**Returns:** either the legend properties when all arguments are `Undefined` or `self`.

**Example:**

```python
scatter.legend(true, 'top-right', 'small')
```


<h3><a name="scatter.legend" href="#scatter.tooltip">#</a> scatter.<b>tooltip</b>(<i>enable=Undefined</i>, <i>contents=Undefined</i>, <i>size=Undefined</i>)</h3>

Set or get the tooltip settings.

**Arguments:**
- `enable` is a Boolean specifying if the tooltip should be enabled or disabled.
- `contents` is either `"all"` or a set of string specifying for which visual channels the data should be shown in the tooltip. It can be some of `x`, `y`, `color`, `opacity`, and `size`.
- `size` is a string specifying the size of the legend. It must be one of `small`, `medium`, or `large`.

**Returns:** either the legend properties when all arguments are `Undefined` or `self`.

**Example:**

```python
scatter.tooltip(true, 'all', 'small')
```


<h3><a name="scatter.zoom" href="#scatter.zoom">#</a> scatter.<b>zoom</b>(<i>target=Undefined</i>, <i>animation=Undefined</i>, <i>padding=Undefined</i>, <i>on_selection=Undefined</i>, <i>on_filter=Undefined</i>)</h3>

Zoom to a set of points.

**Arguments:**
- `to` is a list of point indices or `None`. When set to `None` the camera zoom is reset.
- `animation` defines whether to animate the transition to the new zoom state. This value can either be a Boolean or an Integer specifying the duration of the animation in milliseconds.
- `padding` is the relative padding around the bounding box of the target points. E.g., `0` stands for no padding and `1` stands for a padding that is as wide and tall as the width and height of the points' bounding box.
- `on_selection` if `True` jscatter will automatically zoom to selected points.
- `on_filter` if `True` jscatter will automatically zoom to filtered points.

**Returns:** either the current zoom state (when all arguments are `Undefined`) or `self`.

**Example:**

```python
scatter.zoom([0, 1, 2, 3])
scatter.zoom(None)
scatter.zoom(scatter.selection())
scatter.zoom(to=scatter.selection(), animation=2000, padding=0.1)
```


<h3><a name="scatter.camera" href="#scatter.camera">#</a> scatter.<b>camera</b>(<i>target=Undefined</i>, <i>distance=Undefined</i>, <i>rotation=Undefined</i>, <i>view=Undefined</i>)</h3>

Get or set the camera view.

**Arguments:**
- `target` is a float tuple defining the view center.
- `distance` is a float value defining the distance of the camera from the scatter plot (imagine as a 2D plane in a 3D world).
- `rotation` is a float value defining the rotation in radians.
- `view` is an array-like list of 16 floats defining a view matrix.

**Returns:** either the camera properties when all arguments are `Undefined` or `self`.

**Example:**

```python
scatter.camera(target=[0.5, 0.5])
```


<h3><a name="scatter.mouse" href="#scatter.mouse">#</a> scatter.<b>mouse</b>(<i>mode=Undefined</i>)</h3>

Get or set the mouse mode.

**Arguments:**
- `mode` is either `'panZoom'`, `'lasso'`, or `'rotate'`

**Returns:** either the mouse mode when mode is `Undefined` or `self`.

**Example:**

```python
scatter.mouse(mode='lasso')
```


<h3><a name="scatter.lasso" href="#scatter.lasso">#</a> scatter.<b>lasso</b>(<i>color=Undefined</i>, <i>initiator=Undefined</i>, <i>min_delay=Undefined</i>, <i>min_dist=Undefined</i>, <i>on_long_press=Undefined</i>)</h3>

Get or set the lasso for selecting multiple points.

**Arguments:**
- `color` is a string referring to a Matplotlib-compatible color.
- `initiator` is a Boolean value to specify if the click-based lasso initiator should be enabled or not.
- `min_delay` is an integer specifying the minimal delay in milliseconds before a new lasso point is stored. Higher values will result in more coarse grain lasso polygons but might be more performant. 
- `min_dist` is an integer specifying the minimal distance in pixels that the mouse has to move before a new lasso point is stored. Higher values will result in more coarse grain lasso polygons but might be more performant.
- `on_long_press` is a Boolean value specifying if the lasso should be activated upon a long press.

**Returns:** either the lasso properties when all arguments are `Undefined` or `self`.

**Example:**

```python
scatter.lasso(initiator=True)
```


<h3><a name="scatter.reticle" href="#scatter.reticle">#</a> scatter.<b>reticle</b>(<i>show=Undefined</i>, <i>color=Undefined</i>)</h3>

Get or set the reticle for the point hover interaction.

**Arguments:**
- `show` is a Boolean value to display the reticle when set to `True`.
- `color` is either a string referring to a Matplotlib-compatible color or `'auto'`.

**Returns:** either the reticle properties when all arguments are `Undefined` or `self`.

**Example:**

```python
scatter.reticle(show=True, color="red")
```


<h3><a name="scatter.background" href="#scatter.background">#</a> scatter.<b>background</b>(<i>color=Undefined</i>, <i>image=Undefined</i>)</h3>

Get or set a background color or image.

**Arguments:**
- `color` is a string representing a color compatible with Matplotlib
- `image` is either a URL string pointing to an image or a PIL image understood by [Matplotlib's imshow() method](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.imshow.html)

**Returns:** either the background properties when all arguments are `Undefined` or `self`.

**Example:**

```python
scatter.background(color='black')
scatter.background(color='#000000')
scatter.background(image='https://picsum.photos/640/640?random')
```


<h3><a name="scatter.options" href="#scatter.options">#</a> scatter.<b>options</b>(<i>options=Undefined</i>)</h3>

Get or set other [regl-scatterplot](https://github.com/flekschas/regl-scatterplot) options.

**Arguments:**
- `options` is a dictionary of [regl-scatterplot properties](https://github.com/flekschas/regl-scatterplot/#properties).

**Returns:** either the options when options are `Undefined` or `self`.


<h3><a name="scatter.pixels" href="#scatter.pixels">#</a> scatter.<b>pixels</b>()</h3>

Gets the pixels of the current scatter plot view. Make sure to first download
the pixels first by clicking on the button with the camera icon.

**Returns:** a Numpy array with the pixels in RGBA format.


## Properties

The following is a list of all _settable_ properties of a `Scatter` instance.
You can define those property when creating a `Scatter` instance. For example,
`Scatter(data=df, x='speed', x_scale='log', ...)`.

| Name                       | Type                                                                                     | Default                                        |
| -------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `data`                     | pandas.DataFrame                                                                         | `None`                                         |
| `x`                        | str \| list[float] \| ndarray                                                            | `None`                                         |
| `x_scale`                  | 'linear' \| 'log' \| 'pow' \| tuple[float] \| [LogNorm][lognorm] \| [PowerNorm][pownorm] | `linear`                                       |
| `y`                        | str \| list[float] \| ndarray                                                            | `None`                                         |
| `y_scale`                  | 'linear' \| 'log' \| 'pow' \| tuple[float] \| [LogNorm][lognorm] \| [PowerNorm][pownorm] | `linear`                                       |
| `selection`                | list[int]                                                                                | `[]`                                           |
| `width`                    | int \| 'auto'                                                                            | `'auto'`                                       |
| `height`                   | int                                                                                      | `240`                                          |
| `color`                    | str \| tuple[float] \| list[float]                                                       | `(0, 0, 0, 0.66)`                              |
| `color_selected`           | str \| tuple[float] \| list[float]                                                       | `(0, 0.55, 1, 1)`                              |
| `color_hover`              | str \| tuple[float] \| list[float]                                                       | `(0, 0, 0, 1)`                                 |
| `color_by`                 | str \| list[float \| str]                                                                | `None`                                         |
| `color_map`                | str \| list[str] \| [Colormap][colormap] \| dict \| 'auto'                               | `None`                                         |
| `color_norm`               | tuple[float] \| [Normalize][linnorm]                                                     | `matplotlib.colors.Normalize(0, 1, clip=True)` |
| `color_order`              | list[str \| int] \| 'reverse'                                                            | `None`                                         |
| `opacity`                  | float                                                                                    | `0.66`                                         |
| `opacity_unselected`       | float                                                                                    | `0.5`                                         |
| `opacity_by`               | str \| list[float]                                                                       | `'density'`                                    |
| `opacity_map`              | triple[float] \| list[float] \| dict \| 'auto'                                           | `None`                                         |
| `opacity_norm`             | tuple[float] \| [Normalize][linnorm]                                                     | `matplotlib.colors.Normalize(0, 1, clip=True)` |
| `opacity_order`            | list[str \| int] \| 'reverse'                                                            | `None`                                         |
| `size`                     | int                                                                                      | `3`                                            |
| `size_by`                  | str \| list[int]                                                                         | `None`                                         |
| `size_map`                 | triple[float] \| list[int] \| dict \| 'auto'                                             | `None`                                         |
| `size_norm`                | tuple[float] \| [Normalize][linnorm]                                                     | `matplotlib.colors.Normalize(0, 1, clip=True)` |
| `size_order`               | list[str \| int] \| 'reverse'                                                            | `None`                                         |
| `connect_by`               | str \| list[int]                                                                         | `None`                                         |
| `connect_order`            | str \| list[int]                                                                         | `None`                                         |
| `connection_color`         | str \| tuple[float] \| list[float]                                                       | `(0, 0, 0, 0.1)`                               |
| `connection_color_selected`  | str \| tuple[float] \| list[float]                                                       | `(0, 0.55, 1, 1)`                              |
| `connection_color_hover`   | str \| tuple[float] \| list[float]                                                       | `(0, 0, 0, 0.66)`                              |
| `connection_color_by`      | str \| list[float \| str]                                                                | `None`                                         |
| `connection_color_map`     | str \| list[str] \| [Colormap][colormap] \| dict \| 'auto'                               | `None`                                         |
| `connection_color_norm`    | tuple[float] \| [Normalize][linnorm]                                                     | `matplotlib.colors.Normalize(0, 1, clip=True)` |
| `connection_color_order`   | list[str \| int] \| 'reverse'                                                            | `None`                                         |
| `connection_opacity`       | float                                                                                    | `0.1`                                          |
| `connection_opacity_by`    | str \| list[float]                                                                       | `None`                                         |
| `connection_opacity_map`   | triple[float] \| list[float] \| dict \| 'auto'                                           | `None`                                         |
| `connection_opacity_norm`  | tuple[float] \| [Normalize][linnorm]                                                     | `matplotlib.colors.Normalize(0, 1, clip=True)` |
| `connection_opacity_order` | list[str \| int] \| 'reverse'                                                            | `None`                                         |
| `connection_size`          | int                                                                                      | `2`                                            |
| `connection_size_by`       | str \| list[int]                                                                         | `None`                                         |
| `connection_size_map`      | triple[float] \| list[int] \| dict \| 'auto'                                             | `None`                                         |
| `connection_size_norm`     | tuple[float] \| [Normalize][linnorm]                                                     | `matplotlib.colors.Normalize(0, 1, clip=True)` |
| `connection_size_order`    | list[str \| int] \| 'reverse'                                                            | `None`                                         |
| `axes`                     | bool                                                                                     | `True`                                         |
| `axes_grid`                | bool                                                                                     | `False`                                        |
| `lasso_color`              | str \| tuple[float] \| list[float]                                                       | `(0, 0.666666667, 1, 1)`                       |
| `lasso_initiator`          | bool                                                                                     | `False`                                        |
| `lasso_min_delay`          | int                                                                                      | `10`                                           |
| `lasso_min_dist`           | int                                                                                      | `3`                                            |
| `lasso_on_long_press`      | bool                                                                                     | `True`                                         |
| `reticle`                  | bool                                                                                     | `True`                                         |
| `reticle_color`            | str \| 'auto'                                                                            | `'auto'`                                       |
| `background_color`         | str                                                                                      | `'white'`                                      |
| `background_image`         | str \| array-like or PIL image                                                           | `None`                                         |
| `mouse_mode`               | 'panZoom' \| 'lasso' \| 'rotate'                                                         | `'panZoom'`                                    |
| `camera_target`            | tuple[float]                                                                             | `[0, 0]`                                       |
| `camera_distance`          | float                                                                                    | `1`                                            |
| `camera_rotation`          | float                                                                                    | `0`                                            |
| `camera_view`              | list[float]                                                                              | `None`                                         |
| `zoom_to`                  | list[int]                                                                                | `None`                                         |
| `zoom_animation`           | int                                                                                      | `500`                                          |
| `zoom_on_selection`        | list[float]                                                                              | `0.33`                                         |
| `zoom_on_filter`           | list[float]                                                                              | `False`                                        |
| `zoom_padding`             | list[float]                                                                              | `False`                                        |
| `options`                  | dict                                                                                     | `{}`                                           |

[LogNorm]: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.colors.LogNorm.html
[PowNorm]: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.colors.PowerNorm.html
[LinNorm]: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.colors.Normalize.html
[Colormap]: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.colors.Colormap.html

## Widget

_So sorry, I haven't had a chance to document the widget properties yet._


# Plotting

<h3><a name="plot" href="#plot">#</a> jscatter.<b>plot</b>(<i>x</i>, <i>y</i>, <i>data = None</i>, <i>**kwargs</i>)</h3>

A shorthand function that creates a new scatter instance and immediately returns its widget.

**Arguments:** are the same as of [`Scatter`](#Scatter).

**Returns:** a new scatter widget.

**Examples:**

```python
from jscatter import plot
plot(data=cars, x='speed', y='weight', color='black', opacity_by='density', size=4)
```

# Composing & Linking

<h3><a name="compose" href="#compose">#</a> jscatter.<b>compose</b>(<i>scatters</i>, <i>sync_views = False</i>, <i>sync_selection = False</i>, <i>sync_hover = False</i>, <i></h3>match_by = 'index'</i>,  <i>cols = None</i>, <i>rows = 1</i>, <i>row_height = 320</i>)

A function to compose multiple scatter plot instances in a grid and allow synchronized view, selection, and hover interactions.

**Arguments:**

- `scatters` a list of scatter plot instances
- `sync_views` a Boolean enabling synchronized panning & zooming when set to `True`
- `sync_selection` a Boolean enabling synchronized point selection when set to `True`
- `sync_hover` a Boolean enabling synchronized point hovering when set to `True`
- `match_by` a string or a list of strings referencing a column in the scatters' `data` frame for identifying corresponding data points. When set to `index` corresponding points are associated by their index. The referenced column must hold strings or categorical data.
- `cols` a number specifying the number of columns or `None`. When set to `None` the number of columns is derived from the number of scatters and `rows`.
- `rows` a number specifying the number of rows.
- `row_height` a number specifying the row height in pixels.

**Returns:** a grid of scatter widgets.

**Examples:**

```python
from jscatter import Scatter, compose
from numpy.random import rand

compose(
    [Scatter(x=rand(500), y=rand(500)) for i in range(4)],
    sync_selection=True,
    sync_hover=True,
    rows=2
)
```

<h3><a name="link" href="#link">#</a> jscatter.<b>link</b>(<i>scatters</i>, <i>match_by = 'index'</i>, <i>cols = None</i>, <i>rows = 1</i>, <i>row_height = 320</i>)</h3>

A shorthand function to compose multiple scatter plot instances in a grid and synchronize their view, selection, and hover interactions.

**Arguments:** same as from [`compose()`](#compose)

**Returns:** a grid of linked scatter widgets.

**Examples:**

```python
from jscatter import Scatter, link
from numpy.random import rand
link([Scatter(x=rand(500), y=rand(500)) for i in range(4)], rows=2)
```

# Color Maps

<h3><a name="okabe-ito" href="#okabe-ito">#</a> jscatter.<b>okabe_ito</b></h3>

A colorblind safe categorical color map consisting of eight colors created by Okabe & Ito.

- ![#56B4E9](https://via.placeholder.com/15/56B4E9/000000?text=+) `Sky blue (#56B4E9)`
- ![#E69F00](https://via.placeholder.com/15/E69F00/000000?text=+) `Orange (#E69F00)`
- ![#009E73](https://via.placeholder.com/15/009E73/000000?text=+) `Bluish green (#009E73)`
- ![#F0E442](https://via.placeholder.com/15/F0E442/000000?text=+) `Yellow (#F0E442)`
- ![#0072B2](https://via.placeholder.com/15/0072B2/000000?text=+) `Blue (#0072B2)`
- ![#D55E00](https://via.placeholder.com/15/D55E00/000000?text=+) `Vermillion (#D55E00)`
- ![#CC79A7](https://via.placeholder.com/15/CC79A7/000000?text=+) `Reddish Purple (#CC79A7)`
- ![#000000](https://via.placeholder.com/15/000000/000000?text=+) `Black (#000000)`

<h3><a name="glasbey-light" href="#glasbey-light">#</a> jscatter.<b>glasbey_light</b></h3>

A categorical color map consisting of 256 maximally distinct colors optimized for a _bright_ background. The colors were generated with the fantastic [Colorcet](https://colorcet.holoviz.org) package, which employs an algorithm developed by [Glasbey et al., 2007](https://strathprints.strath.ac.uk/30312/1/colorpaper_2006.pdf).

<h3><a name="glasbey-dark" href="#glasbey-dark">#</a> jscatter.<b>glasbey_dark</b></h3>

A categorical color map consisting of 256 maximally distinct colors optimized for a _dark_ background. The colors were generated with the fantastic [Colorcet](https://colorcet.holoviz.org) package, which employs an algorithm developed by [Glasbey et al., 2007](https://strathprints.strath.ac.uk/30312/1/colorpaper_2006.pdf).
