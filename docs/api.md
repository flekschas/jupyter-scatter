# API Reference

- [Scatter](#scatter)
  - [Methods](#methods)
    - [x()](#scatter.x), [y()](#scatter.x), [xy()](#scatter.xy), and [data()](#scatter.data)
    - [selection()](#scatter.selection) and [filter()](#scatter.filter)
    - [color()](#scatter.color), [opacity()](#scatter.opacity), and [size()](#scatter.size)
    - [connect()](#scatter.connect), [connection_color()](#scatter.connection_color), [connection_opacity()](#scatter.connection_opacity), and [connection_size()](#scatter.connection_size)
    - [axes()](#scatter.axes), [legend()](#scatter.legend), and [annotations()](#scatter.annotations)
    - [tooltip()](#scatter.tooltip) and [show_tooltip()](#scatter.show_tooltip)
    - [zoom()](#scatter.zoom) and [camera()](#scatter.camera)
    - [lasso()](#scatter.lasso), [reticle()](#scatter.reticle), and [mouse()](#scatter.mouse),
    - [background()](#scatter.background) and [options()](#scatter.options)
  - [Properties](#properties)
  - [Widget](#widget)
- [Plotting Shorthand](#plotting)
- [Composing \& Linking](#composing-linking)
- [Color Maps](#color-maps)


## Scatter

### Scatter(_x_, _y_, _data=None_, _\*\*kwargs_) {#Scatter}

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


## Methods {#methods}

### scatter.x(_x=Undefined_, _scale=Undefined_, _\*\*kwargs_) {#scatter.x}

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


### scatter.y(_y=Undefined_, _scale=Undefined_, _\*\*kwargs_) {#scatter.y}

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


### scatter.xy(_x=Undefined_, _y=Undefined_, _x_scale=Undefined_, _y_scale=Undefined_, _\*\*kwargs_) {#scatter.xy}

Get or set the x and y coordinate. This is just a convenience function to animate a change in the x and y coordinate at the same time.

**Arguments:**

- `x` is either an array-like list of coordinates or a string referencing a column in `data`.
- `y` is either an array-like list of coordinates or a string referencing a column in `data`.
- `x_scale` is either a string (`linear`, `time`, `log`, `pow`), a tuple defining the value range that's map to the extent of the scatter plot, or an instance of `matplotlib.colors.LogNorm` or `matplotlib.colors.PowerNorm`.
- `y_scale` is either a string (`linear`, `time`, `log`, `pow`), a tuple defining the value range that's map to the extent of the scatter plot, or an instance of `matplotlib.colors.LogNorm` or `matplotlib.colors.PowerNorm`.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the x and y coordinate when x and y are `Undefined` or `self`.

**Examples:**

```python
scatter.xy('size', 'speed') # Mirror plot along the diagonal
```


### scatter.data(_data=Undefined_, _use_index=Undefined_, _\*\*kwargs_) {#scatter.data}

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


### scatter.selection(_point_idxs=Undefined_) {#scatter.selection}

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


### scatter.filter(_point_idxs=Undefined_) {#scatter.filter}

Get or set the filtered points. When filtering down to a set of points, all other points will be hidden from the view.

**Arguments:**

- `point_idxs` is a list or an array-like object of point indices or `None`.

**Returns:** either the currently filtered point indices when `point_idxs` is `Undefined` or `self`.

**Examples:**

```python
scatter.filter(cars.query('speed < 50').index)
scatter.filter(None) # To unset filter
```


### scatter.color(_default=Undefined_, _selected=Undefined_, _hover=Undefined_, _by=Undefined_, _map=Undefined_, _norm=Undefined_, _order=Undefined_, _labeling=Undefined_, _\*\*kwargs_) {#scatter.color}

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


### scatter.opacity(_default=Undefined_, _unselected=Undefined_, _by=Undefined_, _map=Undefined_, _norm=Undefined_, _order=Undefined_, _labeling=Undefined_, _\*\*kwargs_) {#scatter.opacity}

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


### scatter.size(_default=Undefined_, _by=Undefined_, _map=Undefined_, _norm=Undefined_, _order=Undefined_, _labeling=Undefined_, _\*\*kwargs_) {#scatter.size}

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


### scatter.connect(_by=Undefined_, _order=Undefined_, _\*\*kwargs_) {#scatter.connect}

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


### scatter.connection_color(_default=Undefined_, _selected=Undefined_, _hover=Undefined_, _by=Undefined_, _map=Undefined_, _norm=Undefined_, _order=Undefined_, _labeling=Undefined_, _\*\*kwargs_) {#scatter.connection_color}

Get or set the point connection color. This function behaves identical to [scatter.color()][#scatter.color].


### scatter.connection_opacity(_default=Undefined_, _by=Undefined_, _map=Undefined_, _norm=Undefined_, _order=Undefined_, _labeling=Undefined_, _\*\*kwargs_) {#scatter.connection_opacity}

Get or set the point connection opacity. This function behaves identical to [scatter.opacity()][#scatter.opacity].


### scatter.connection_size(_default=Undefined_, _by=Undefined_, _map=Undefined_, _norm=Undefined_, _order=Undefined_, _labeling=Undefined_, _\*\*kwargs_) {#scatter.connection_size}

Get or set the point connection size. This function behaves identical to [scatter.size()]#[scatter.size].


### scatter.axes(_axes=Undefined_, _grid=Undefined_, _labels=Undefined_) {#scatter.axes}

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


### scatter.legend(_legend=Undefined_, _position=Undefined_, _size=Undefined_) {#scatter.legend}

Set or get the legend settings.

**Arguments:**
- `legend` is a Boolean specifying if the legend should be shown or not.
- `position` is a string specifying the legend position. It must be one of `top`, `left`, `right`, `bottom`, `top-left`, `top-right`, `bottom-left`, `bottom-right`, or `center`.
- `size` is a string specifying the size of the legend. It must be one of `small`, `medium`, or `large`.

**Returns:** either the legend properties when all arguments are `Undefined` or `self`.

**Example:**

```python
scatter.legend(True, 'top-right', 'small')
```

### scatter.annotations(_annotations=Undefined_) {#scatter.annotations}

Set or get annotations.

**Arguments:**
- `annotations` is a list of annotations (`Line`, `HLine`, `VLine`, or `Rect`)

**Returns:** either the annotation properties when all arguments are `Undefined` or `self`.

**Example:**

```python
from jscatter import HLine, VLine
scatter.annotations([HLine(42), VLine(42)])
```

### scatter.tooltip(_enable=Undefined_, _properties=Undefined_, _histograms=Undefined_, _histograms_bins=Undefined_, _histograms_ranges=Undefined_, _histograms_size=Undefined_, _preview=Undefined_, _preview_type=Undefined_, _preview_text_lines=Undefined_, _preview_image_background_color=Undefined_, _preview_image_position=Undefined_, _preview_image_size=Undefined_, _preview_audio_length=Undefined_, _preview_audio_loop=Undefined_, _preview_audio_controls=Undefined_, _size=Undefined_) {#scatter.tooltip}

Set or get the tooltip settings.

**Arguments:**
- `enable` is a Boolean specifying if the tooltip should be enabled or disabled.

- `properties` is a list of string specifying for which visual or data properties to show in the tooltip. The visual properties can be some of `x`, `y`, `color`, `opacity`, and `size`. Note that visual properties are only shown if they are actually used to data properties. To reference other data properties, specify a column of the bound DataFrame by its name.

- `histograms` is a Boolean specifying if the tooltip should show histograms of the properties

- `histograms_bins` is either an Integer specifying the number of bins of all numerical histograms or a dictionary of property-specific number of bins. The default is `20`.

- `histograms_ranges` is either a tuple of the lower and upper range of all bins or a dictionary of property-specific lower upper bin ranges. The default is `(min(), max())`.

- `histograms_size` is a string specifying the size of the histograms. It must be one of `small`, `medium`, or `large`. The default is `"small"`.

- `preview` is a string referencing a column name of the bound DataFrame that contains preview data. Currently three data types are supported: plain text, URLs referencing images, and URLs referencing audio.

- `preview_type` is a string specifying the media type of the preview. This can be one of `"text"`, `"image"`, or `"audio"`. The default is `"text"`.

- `preview_text_lines` is an integer specifying the maximum number of lines for text previews that should be displayed. Text that exceeds defined limit will be truncated with an ellipsis. By default, the line limit is set to `None` to be disabled.

- `preview_image_background_color` is a string specifying the background color for image previews. By default, the value is `None`, which means that image preview has a transparent background. In this case and if `preview_image_size` is set to `"contain"` and your image does not perfectly cover the preview area, you will see the tooltip's background color.

- `preview_image_position` is a string specifying the image position of image previews. This can be one of `"top"`, `"bottom"`, `"left"`, `"right"`, or `"center"`. The default value is `"center"`.
  
  See https://developer.mozilla.org/en-US/docs/Web/CSS/background-position for details on the behavior.

- `preview_image_size` is a string specifying the size of the image in the context of the preview area. This can be one of `"cover"` or `"contain"` and is set to `"contain"` by default.

  See https://developer.mozilla.org/en-US/docs/Web/CSS/background-size for details on the behavior.

- `preview_audio_length` is an integer specifying the number of seconds of an audio preview that should be played. By default (`None`), the audio file is played from the start to the end.

- `preview_audio_loop` is a Boolean specifying if the audio preview is indefinitely looped for the duration the tooltip is shown.

  See https://developer.mozilla.org/en-US/docs/Web/HTML/Element/audio#loop for details on the behavior.

- `preview_audio_controls` is a Boolean specifying if the audio preview will include controls. While you cannot interact with the controls (as the tooltip disappears upon leaving a point), the controls show the progression and length of the played audio.

  See https://developer.mozilla.org/en-US/docs/Web/HTML/Element/audio#controls for details on the behavior.

- `size` is a string specifying the size of the tooltip. It must be one of `small`, `medium`, or `large`. The default is `"small"`.

**Returns:** either the legend properties when all arguments are `Undefined` or `self`.

**Example:**

```python
scatter.tooltip(
  enable=True,
  properties=["color", "opacity", "effect_size"],
  histograms=True,
  histograms_bins=12,
  histograms_ranges={"effect_size": (0.5, 1.5)},
  histograms_width="medium",
  preview="image_url",
  preview_type="image",
  preview_image_background_color="black",
  preview_image_position="center",
  preview_image_size="cover",
  size="small",
)
```


### scatter.show_tooltip(_point_idx_) {#scatter.show_tooltip}

Programmatically show a tooltip for a point.

::: info
The tooltip is not permanent and will go away as soon as you mouse over some
other points in the plot.
:::

::: warning
If the widget has not been instantiated yet or the tooltip has not been
activated via `scatter.tooltip(True)`, this method is a noop.
:::

**Arguments:**
- `point_idx` is a point index.

**Example:**

```python
scatter.show_tooltip(42)
```


### scatter.zoom(_to=Undefined_, _animation=Undefined_, _padding=Undefined_, _on_selection=Undefined_, _on_filter=Undefined_) {#scatter.zoom}

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


### scatter.camera(_target=Undefined_, _distance=Undefined_, _rotation=Undefined_, _view=Undefined_) {#scatter.camera}

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


### scatter.mouse(_mode=Undefined_) {#scatter.mouse}

Get or set the mouse mode.

**Arguments:**
- `mode` is either `'panZoom'`, `'lasso'`, or `'rotate'`

**Returns:** either the mouse mode when mode is `Undefined` or `self`.

**Example:**

```python
scatter.mouse(mode='lasso')
```


### scatter.lasso(_color=Undefined_, _initiator=Undefined_, _min_delay=Undefined_, _min_dist=Undefined_, _on_long_press=Undefined_) {#scatter.lasso}

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


### scatter.reticle(_show=Undefined_, _color=Undefined_) {#scatter.reticle}

Get or set the reticle for the point hover interaction.

**Arguments:**
- `show` is a Boolean value to display the reticle when set to `True`.
- `color` is either a string referring to a Matplotlib-compatible color or `'auto'`.

**Returns:** either the reticle properties when all arguments are `Undefined` or `self`.

**Example:**

```python
scatter.reticle(show=True, color="red")
```


### scatter.background(_color=Undefined_, _image=Undefined_) {#scatter.background}

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


### scatter.options(_transition_points=Undefined_, _transition_points_duration=Undefined_, _regl_scatterplot_options=Undefined_) {#scatter.options}

Get or set other Jupyter Scatter and [regl-scatterplot](https://github.com/flekschas/regl-scatterplot) options.

**Arguments:**
- `transition_points` is a Boolean value to enable or disable the potential animated transitioning of points as their coordinates update. If `False`, points will never be animated.
- `transition_points_duration` is an Integer value determining the time of the animated point transition in milliseconds. The default value is `3000`.
- `regl_scatterplot_options` is a dictionary of [regl-scatterplot properties](https://github.com/flekschas/regl-scatterplot/#properties).

**Returns:** either the options when options are `Undefined` or `self`.


### scatter.pixels() {#scatter.pixels}

Gets the pixels of the current scatter plot view. Make sure to first download
the pixels first by clicking on the button with the camera icon.

**Returns:** a Numpy array with the pixels in RGBA format.


## Properties {#properties}

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
| `connection_color_selected`| str \| tuple[float] \| list[float]                                                       | `(0, 0.55, 1, 1)`                              |
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

## Widget {#widget}

The widget (`scatter.widget`) has the following properties, which you can think of as the view model of Jupyter Scatter.

::: warning
While you can adjust these properties directly, the [`Scatter` methods](#methods) are the idiomatic and recommended way to set widget properties.
:::

| Name                                     | Type <div style="min-width:250px"/>                                                                                                                                  | Default <div style="min-width:180px"/>   | Allow None | Read Only | Note <div style="min-width:320px"/> |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ---------- | --------- | ------------- |
| `dom_element_id`                         | str                                                                                                                                                                  |                                            |            | `True`    | For debugging |
| `data`                                   | 2D numerical array                                                                                                                                                   |                                            |            |           |               |
| `prevent_filter_reset`                   | bool                                                                                                                                                                 | `False`                                    |            |           |               |
| `selection`                              | int[]                                                                                                                                                                |                                            | `True`     |           | Point indices |
| `filter`                                 | int[]                                                                                                                                                                |                                            | `True`     |           | Point indices |
| `hovering`                               | int                                                                                                                                                                  |                                            | `True`     |           |               |
| `x_title`                                | str                                                                                                                                                                  |                                            | `True`     |           |               |
| `y_title`                                | str                                                                                                                                                                  |                                            | `True`     |           |               |
| `color_title`                            | str                                                                                                                                                                  |                                            | `True`     |           |               |
| `opacity_title`                          | str                                                                                                                                                                  |                                            | `True`     |           |               |
| `size_title`                             | str                                                                                                                                                                  |                                            | `True`     |           |               |
| `x_scale`                                | str                                                                                                                                                                  |                                            | `True`     |           |               |
| `y_scale`                                | str                                                                                                                                                                  |                                            | `True`     |           |               |
| `color_scale`                            | str                                                                                                                                                                  |                                            | `True`     |           |               |
| `opacity_scale`                          | str                                                                                                                                                                  |                                            | `True`     |           |               |
| `size_scale`                             | str                                                                                                                                                                  |                                            | `True`     |           |               |
| `x_domain`                               | [float, float]                                                                                                                                                       |                                            |            |           |               |
| `y_domain`                               | [float, float]                                                                                                                                                       |                                            |            |           |               |
| `x_scale_domain`                         | [float, float]                                                                                                                                                       |                                            |            |           |               |
| `y_scale_domain`                         | [float, float]                                                                                                                                                       |                                            |            |           |               |
| `color_domain`                           | [float, float] \| {}                                                                                                                                                 |                                            | `True`     |           |               |
| `opacity_domain`                         | [float, float] \| {}                                                                                                                                                 |                                            | `True`     |           |               |
| `size_domain`                            | [float, float] \| {}                                                                                                                                                 |                                            | `True`     |           |               |
| `x_histogram`                            | float[]                                                                                                                                                              |                                            | `True`     |           |               |
| `y_histogram`                            | float[]                                                                                                                                                              |                                            | `True`     |           |               |
| `color_histogram`                        | float[]                                                                                                                                                              |                                            | `True`     |           |               |
| `opacity_histogram`                      | float[]                                                                                                                                                              |                                            | `True`     |           |               |
| `size_histogram`                         | float[]                                                                                                                                                              |                                            | `True`     |           |               |
| `x_histogram_range`                      | [float, float]                                                                                                                                                       |                                            | `True`     |           |               |
| `y_histogram_range`                      | [float, float]                                                                                                                                                       |                                            | `True`     |           |               |
| `color_histogram_range`                  | [float, float]                                                                                                                                                       |                                            | `True`     |           |               |
| `opacity_histogram_range`                | [float, float]                                                                                                                                                       |                                            | `True`     |           |               |
| `size_histogram_range`                   | [float, float]                                                                                                                                                       |                                            | `True`     |           |               |
| `camera_target`                          | [float, float]                                                                                                                                                       |                                            |            |           |               |
| `camera_distance`                        | float                                                                                                                                                                |                                            |            |           |               |
| `camera_rotation`                        | float                                                                                                                                                                |                                            |            |           |               |
| `camera_view`                            | float[]                                                                                                                                                              |                                            | `True`     |           | View matrix   |
| `zoom_to`                                | int[]                                                                                                                                                                |                                            |            |           | Point indices |
| `zoom_animation`                         | int                                                                                                                                                                  | `1000`                                     |            |           | Animation time in milliseconds |
| `zoom_padding`                           | float                                                                                                                                                                | `0.333`                                    |            |           | Zoom padding relative to the bounding box of the points to zoom to |
| `zoom_on_selection`                      | bool                                                                                                                                                                 | `False`                                    |            |           | If `True` zoom to selected points automatically |
| `zoom_on_filter`                         | bool                                                                                                                                                                 | `False`                                    |            |           | If `True` zoom to filtered points automatically |
| `mouse_mode`                             | `"panZoom"` \| `"lasso"` \| `"rotate"`                                                                                                                               | `"panZoom"`                                |            |           |               |
| `lasso_initiator`                        | bool                                                                                                                                                                 | `False`                                    |            |           |               |
| `lasso_on_long_press`                    | bool                                                                                                                                                                 | `True`                                     |            |           |               |
| `axes`                                   | bool                                                                                                                                                                 | `True`                                     |            |           |               |
| `axes_grid`                              | bool                                                                                                                                                                 | `False`                                    |            |           |               |
| `axes_color`                             | [float, float, float, float]                                                                                                                                         |                                            |            |           | RGBA          |
| `axes_labels`                            | bool \| str[]                                                                                                                                                        | `False`                                    |            |           |               |
| `legend`                                 | bool                                                                                                                                                                 | `False`                                    |            |           |               |
| `legend_position`                        | `"top"`<br/>\| `"top-right"`<br/>\| `"top-left"`<br/>\| `"bottom"`<br/>\| `"bottom-right"`<br/>\| `"bottom-left"`<br/>\| `"left"`<br/>\| `"right"`<br/>\| `"center"` | `"top-left"`                               |            |           |               |
| `legend_size`                            | `"small"` \| `"medium"` \| `"large"`                                                                                                                                 | `"small"`                                  |            |           |               |
| `legend_color`                           | [float, float, float, float]                                                                                                                                         |                                            |            |           | RGBA          |
| `legend_encoding`                        | {}                                                                                                                                                                   |                                            |            |           |               |
| `tooltip_enable`                         | bool                                                                                                                                                                 | `False`                                    |            |           | Why is this property not just called `tooltip` you might wonder? Ipywidgets seem to internally use this property, which prevents other widgets from using it unfortunately. |
| `tooltip_size`                           | `"small"` \| `"medium"` \| `"large"`                                                                                                                                 | `"small"`                                  |            |           |               |
| `tooltip_color`                          | [float, float, float, float]                                                                                                                                         |                                            |            |           | RGBA          |
| `tooltip_properties`                     | str[]                                                                                                                                                                | `['x', 'y', 'color', 'opacity', 'size']`   |            |           |               |
| `tooltip_properties_non_visual_info`     | {}                                                                                                                                                                   |                                            |            |           |               |
| `tooltip_histograms`                     | bool                                                                                                                                                                 | `True`                                     |            |           |               |
| `tooltip_histograms_ranges`              | dict                                                                                                                                                                 | `True`                                     |            |           |               |
| `tooltip_histograms_size`                | `"small"` \| `"medium"` \| `"large"`                                                                                                                                 | `"small"`                                  |            |           |               |
| `tooltip_preview`                        | str                                                                                                                                                                  |                                            | `True`     |           |               |
| `tooltip_preview_type`                   | `"text"` \| `"image"` \| `"audio"`                                                                                                                                   | `"text"`                                   |            |           |               |
| `tooltip_preview_text_lines`             | int                                                                                                                                                                  | `3`                                        | `True`     |           |               |
| `tooltip_preview_image_background_color` | `"auto"` \| str                                                                                                                                                      | `"auto"`                                   |            |           |               |
| `tooltip_preview_image_position`         | `"top"`<br/>\| `"left"`<br/>\| `"right"`<br/>\| `"bottom"`<br/>\| `"center"`                                                                                         | `"center"`                                 | `True`     |           |               |
| `tooltip_preview_image_size`             | `"contain"` \| `"cover"`                                                                                                                                             | `"contain"`                                | `True`     |           |               |
| `tooltip_preview_audio_length`           | int                                                                                                                                                                  | `None`                                     | `True`     |           |               |
| `tooltip_preview_audio_loop`             | bool                                                                                                                                                                 | `False`                                    |            |           |               |
| `tooltip_preview_audio_controls`         | bool                                                                                                                                                                 | `True`                                     |            |           |               |
| `color`                                  | str<br/>\| str[]<br/>\| [float, float, float, float]<br/>\| [float, float, float, float][]                                                                           | `[0, 0, 0, 0.66]` or<br/>`[1, 1, 1, 0.66]` |            |           | Default value depends on the luminance of the background color. |
| `color_selected`                         | str<br/>\| str[]<br/>\| [float, float, float, float]<br/>\| [float, float, float, float][]                                                                           | `[0, 0.55, 1, 1]`                          |            |           |               |
| `color_hover`                            | str<br/>\| str[]<br/>\| [float, float, float, float]<br/>\| [float, float, float, float][]                                                                           | `[0, 0, 0, 1]` or<br/>`[1, 1, 1, 1]`       |            |           | Default value depends on the luminance of the background color. |
| `color_by`                               | `"valueA"` \| `"valueB"`                                                                                                                                             | `None`                                     | `True`     |           |               |
| `opacity`                                | float \| float[]                                                                                                                                                     | `0.66`                                     |            |           |               |
| `opacity_unselected`                     | float \| float[]                                                                                                                                                     | `0.5`                                      |            |           |               |
| `opacity_by`                             | `"valueA"` \| `"valueB"` \| `"density"`                                                                                                                              | `"density"`                                | `True`     |           |               |
| `size`                                   | int \| int[] \| float \| float[]                                                                                                                                     | `3`                                        |            |           |               |
| `size_by`                                | `"valueA"` \| `"valueB"`                                                                                                                                             | `None`                                     | `True`     |           |               |
| `connect`                                | bool                                                                                                                                                                 | `False`                                    |            |           |               |
| `connection_color`                       | str<br/>\| str[]<br/>\| [float, float, float, float]<br/>\| [float, float, float, float][]                                                                           | `[0, 0, 0, 0.1]` or<br/>`[1, 1, 1, 0.1]`   |            |           | Default value depends on the luminance of the background color. |
| `connection_color_by`                    | `"valueA"` \| `"valueB"` \| `"segment"`                                                                                                                            | `None`                                     | `True`     |           | Default value depends on the luminance of the background color. |
| `connection_color_selected`              | str<br/>\| str[]<br/>\| [float, float, float, float]<br/>\| [float, float, float, float][]                                                                           | `[0, 0.55, 1, 1]`                          |            |           |               |
| `connection_color_hover`                 | str<br/>\| str[]<br/>\| [float, float, float, float]<br/>\| [float, float, float, float][]                                                                           | `[0, 0, 0, 0.66]` or<br/>`[1, 1, 1, 0.66]` |            |           | Default value depends on the luminance of the background color. |
| `connection_opacity`                     | float \| float[]                                                                                                                                                     | `0.1`                                      |            |           |               |
| `connection_opacity_by`                  | `"valueA"` \| `"valueB"` \| `"segment"`                                                                                                                              | `None`                                     | `True`     |           |               |
| `connection_size`                        | int \| int[] \| float \| float[]                                                                                                                                     | `2`                                        |            |           |               |
| `connection_size_by`                     | `"valueA"` \| `"valueB"` \| `"segment"`                                                                                                                              | `None`                                     | `True`     |           |               |
| `width`                                  | int \| `"auto"`                                                                                                                                                      | `"auto"`                                   |            |           |               |
| `height`                                 | int                                                                                                                                                                  | `240`                                      |            |           |               |
| `background_color`                       | str \| [float, float, float, float]                                                                                                                                  | `"white"`                                  |            |           |               |
| `background_image`                       | str                                                                                                                                                                  | `None`                                     | `True`     |           |               |
| `lasso_color`                            | str \| [float, float, float, float]                                                                                                                                  | `[0, 0.666666667, 1, 1]`                   |            |           |               |
| `lasso_min_delay`                        | int                                                                                                                                                                  | `10`                                       |            |           |               |
| `lasso_min_dist`                         | float                                                                                                                                                                | `3`                                        |            |           |               |
| `reticle`                                | bool                                                                                                                                                                 | `True`                                     |            |           |               |
| `reticle_color`                          | str<br/>\| [float, float, float, float]<br/>\| `"auto"`                                                                                                              | `"auto"`                                   |            |           |               |
| `other_options`                          | dict                                                                                                                                                                 | `{}`                                       |            |           | For setting other [regl-scatterplot properties](https://github.com/flekschas/regl-scatterplot/?tab=readme-ov-file#properties). Note that whatever is defined in options will be overwritten by the short-hand options |
| `view_reset`                             | bool                                                                                                                                                                 | `False`                                    |            |           |               |
| `view_download`                          | bool                                                                                                                                                                 | `None`                                     | `True`     |           |               |
| `view_data`                              | int[]                                                                                                                                                                | `None`                                     | `True`     | `True`    | [Uint8ClampedArray](https://developer.mozilla.org/en-US/docs/Web/API/ImageData/data) |
| `view_shape`                             | [int, int]                                                                                                                                                           | `None`                                     | `True`     | `True`    |               |
| `view_sync`                              | str                                                                                                                                                                  | `None`                                     | `True`     |           | For synchronyzing view changes across scatter plot instances |

## Plotting Shorthand {#plotting}

### plot(_x=Undefined_, _y=Undefined_, _data=Undefined_, _\*\*kwargs_) {#plot}

A shorthand function that creates a new scatter instance and immediately returns its widget.

**Arguments:** are the same as of [`Scatter`](#Scatter).

**Returns:** a new scatter widget.

**Examples:**

```python
from jscatter import plot
plot(data=cars, x='speed', y='weight', color='black', opacity_by='density', size=4)
```

## Composing & Linking

### compose(_scatters_, _sync_views=False_, _sync_selection=False_, _sync_hover=False_, _match_by="index"_, _cols=None_, _rows=1_, _row_height=320_) {#compose}

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


### link(_scatters_, _match_by="index"_, _cols=None_, _rows=1_, _row_height=320_) {#link}

A shorthand function to compose multiple scatter plot instances in a grid and synchronize their view, selection, and hover interactions.

**Arguments:** same as from [`compose()`](#compose)

**Returns:** a grid of linked scatter widgets.

**Examples:**

```python
from jscatter import Scatter, link
from numpy.random import rand
link([Scatter(x=rand(500), y=rand(500)) for i in range(4)], rows=2)
```

## Color Maps

### okabe_ito {#okabe-ito}

A colorblind safe categorical color map consisting of eight colors created by Okabe & Ito.

- ![#56B4E9](https://via.placeholder.com/15/56B4E9/000000?text=+) `Sky blue (#56B4E9)`
- ![#E69F00](https://via.placeholder.com/15/E69F00/000000?text=+) `Orange (#E69F00)`
- ![#009E73](https://via.placeholder.com/15/009E73/000000?text=+) `Bluish green (#009E73)`
- ![#F0E442](https://via.placeholder.com/15/F0E442/000000?text=+) `Yellow (#F0E442)`
- ![#0072B2](https://via.placeholder.com/15/0072B2/000000?text=+) `Blue (#0072B2)`
- ![#D55E00](https://via.placeholder.com/15/D55E00/000000?text=+) `Vermillion (#D55E00)`
- ![#CC79A7](https://via.placeholder.com/15/CC79A7/000000?text=+) `Reddish Purple (#CC79A7)`
- ![#000000](https://via.placeholder.com/15/000000/000000?text=+) `Black (#000000)`

### glasbey_light {#glasbey-light}

A categorical color map consisting of 256 maximally distinct colors optimized for a _bright_ background. The colors were generated with the fantastic [Colorcet](https://colorcet.holoviz.org) package, which employs an algorithm developed by [Glasbey et al., 2007](https://strathprints.strath.ac.uk/30312/1/colorpaper_2006.pdf).

### glasbey_dark {#glasbey-dark}

A categorical color map consisting of 256 maximally distinct colors optimized for a _dark_ background. The colors were generated with the fantastic [Colorcet](https://colorcet.holoviz.org) package, which employs an algorithm developed by [Glasbey et al., 2007](https://strathprints.strath.ac.uk/30312/1/colorpaper_2006.pdf).
