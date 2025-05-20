# API Reference

- [Scatter](#scatter)
  - [Methods](#methods)
    - [x()](#scatter.x), [y()](#scatter.x), [xy()](#scatter.xy), and [data()](#scatter.data)
    - [selection()](#scatter.selection) and [filter()](#scatter.filter)
    - [color()](#scatter.color), [opacity()](#scatter.opacity), and [size()](#scatter.size)
    - [connect()](#scatter.connect), [connection_color()](#scatter.connection_color), [connection_opacity()](#scatter.connection_opacity), and [connection_size()](#scatter.connection_size)
    - [axes()](#scatter.axes), [legend()](#scatter.legend), [label()](#scatter.label), and [annotations()](#scatter.annotations)
    - [tooltip()](#scatter.tooltip) and [show_tooltip()](#scatter.show_tooltip)
    - [zoom()](#scatter.zoom) and [camera()](#scatter.camera)
    - [lasso()](#scatter.lasso), [reticle()](#scatter.reticle), and [mouse()](#scatter.mouse),
    - [background()](#scatter.background) and [options()](#scatter.options)
  - [Properties](#properties)
  - [Widget](#widget)
- [Plotting Shorthand](#plotting)
- [Composing \& Linking](#composing-linking)
- [Color Maps](#color-maps)
- [Annotations](#annotations)
- [LabelPlacement](#labelplacement)


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

### scatter.show(_buttons=Undefined_) {#scatter.show}

Show the scatter plot widget.

**Arguments:**
- `buttons`: The buttons to show in the widget. Can be one of the following:
  - `"pan_zoom"`: Button to activate the pan and zoom mode.
  - `"lasso"`: Button to activate the lasso mode.
  - `"full_screen"`: Button to enter full screen mode.
  - `"save"`: Button to save the current view in `scatter.widget.view_data`.
  - `"download"`: Button to download the current view as a PNG image.
  - `"reset"`: Button to reset the view.
  - `"divider"`: Not a button, but a divider between buttons.

**Returns:** either the x coordinate when x is `Undefined` or `self`.

**Examples:**

```python
# Show the widget with all buttons
scatter.show()

# Show the widget with only a subset of buttons
scatter.show(['full_screen', 'download', 'reset'])
```

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


### scatter.data(_data=Undefined_, _use_index=Undefined_, _reset_scales=False_, _zoom_view=False_, _animate=False_, _\*\*kwargs_) {#scatter.data}

Get or set the referenced Pandas DataFrame.

**Arguments:**

- `data` is a Pandas DataFrame.
- `use_index` is a Boolean value indicating if the data frame's index should be used for referencing point by the `selection()` and `filter()` methods instead of the row index.
- `reset_scales` is a Boolean value indicating whether all scales (and norms) will be reset to the extend of the the new data.
- `zoom_view` is a Boolean value indicating if the view will zoom to the data extent.
- `animate` is a Boolean value indicating if the points will transition smoothly. However, animated point transitions are only supported if the number of points remain the same, and if `reset_scales` is `False`. If `zoom_view` is `True`, the view will also transition smoothly.
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


### scatter.size(_default=Undefined_, _by=Undefined_, _map=Undefined_, _norm=Undefined_, _order=Undefined_, _labeling=Undefined_, _scale_function=Undefined_, _\*\*kwargs_) {#scatter.size}

Get or set the point size.

**Arguments:**

- `default` is a valid matplotlib color.
- `by` is either an array-like list of values or a string referencing a column in `data`.
- `map` is either a triple specifying an `np.linspace(*map)`, a list of sizes, a dictionary of category-size pairs, or `auto` (to let jscatter choose a default size map).
- `norm` is either a tuple defining a value range that's map to `[0, 1]` with `matplotlib.colors.Normalize` or a [matplotlib normalizer](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.colors.Normalize.html).
- `order` is either a list of values (for categorical size encoding) or `reverse` to reverse the size map.
- `labeling` is either a tuple of three strings specyfing a label for the minimum value, maximum value, and variable that the size encodes or a dictionary of the form `{'minValue': 'label', 'maxValue': 'label', 'variable': 'label'}`. The specified labels are only used for continuous size encoding and are displayed together with the legend.
- `scale_function` is the function used for adjusting the size of points when zooming in. It can either be `asinh`, `linear`, or `constant`. The default is `asinh`. `constant` is a special case that does not scale the size of points when zooming in.
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

### scatter.label(_legend=Undefined_, _position=Undefined_, _size=Undefined_) {#scatter.label}

Set or get the label settings.

**Arguments:**
- `by` is a string or list of strings referencing columns in `data`. The
  columns should define a hierarchy of points by which you want to
  label these points. When set to `None`, labeling is turned off.

  To display individual point labels (where each row gets its own label),
  append an exclamation mark to the column name. For instance, `"city!"`
  would indicate that each value in this column should be treated as a
  unique label rather than grouping identical values together.

  Note: Currently only one column can be marked with an exclamation mark.
  If multiple columns are marked, only the first one is treated as
  containing point labels.
- `font` is a font or list of fonts for rendering the labels. A list of fonts
  must match `by` in terms of the length. A dict can map specific
  label types or label values to fonts.
- `color` is a single, list or dict of colors for rendering the labels. A list of
  colors must match `by` in terms of the length. A dict of colors must
  define a color for every unique label. By default, the color is set
  to `"auto"` meaning a default color is assigned based on the
  background color.
- `size` is the font size(s) for label text. Can be a single integer for uniform
  size, a list matching the length of `by` for hierarchical sizes, or
  a dictionary mapping specific label types or values to sizes.
- `shadow_color` is The outline color for rendering the labels. By default, the
  color is set to `"auto"` meaning a default color is assigned based on the
  background color.
- `importance` is the column name containing importance values that determine
  which labels to prioritize when there are conflicts. If not specified, all
  labels have equal importance.
- `importance_aggregation` is the method used to aggregate importance values
  when multiple points share the same label. Can be one of `'min'`, `'mean'`,
  `'median'`, `'max'`, `'sum'`. Default is `'mean'`.
- `max_number` is the maximum number of labels per tile. Controls label density
  by limiting how many labels can appear in a given region. Default is `100`.
- `align` is the label alignment relative to the labeled point or group. Can be
  one of `'center'`, `'top-left'`, `'top'`, `'top-right'`, `'left'`, `'right'`,
  `'bottom-left'`, `'bottom'`, `'bottom-right'`. Default is `'center'`.
- `offset` is the x and y offset of the label from the center of the bounding
  box of points it's labeling. Note, this only has an effect when `align`
  is not `'center'`.
- `scale_function` is the scale function by which the text size is adjusted when
  zooming in. Can be one of:
  - `'asinh'`: Scales labels by the inverse hyperbolic sine, initially
    increasing linearly but quickly plateauing (default).
  - `'constant'`: No scaling with zoom, labels maintain the same size.
- `zoom_range`: The range at which labels of a specific type (as specified with
  `by`) are allowed to appear. The zoom range is a tuple of zoom levels,
  where `zoom_scale == 2 ** zoom_level`. Defaults to (-∞, ∞) for all
  labels. Can be specified as:
  - Single tuple: Applied to all label types
  - List of tuples: One range per label type in `by`
  - Dict: Maps label types or specific labels to their zoom ranges
- `hierarchical` is a flag. If `True`, the label types specified by `by` are
  treated as hierarchical. This affects label priority, with labels having a
  lower hierarchical index (earlier in the `by` list) displayed first when there
  are conflicts. Default is `False`.
- `exclude` is a list of string that specifies which labels should be excluded.
  Can contain:
  - Column names (e.g., `'country'`) to exclude an entire category
  - Column-value pairs (e.g., `'country:Germany'`) to exclude specific labels
- `positioning` is the method for determining label position. Options are:
  - `'highest_density'` (default): Position label at the highest density point
  - `'center_of_mass'`: Position label at the center of mass of all points
  - `'largest_cluster'`: Position label at the center of the largest cluster
- `target_aspect_ratio` is a float in `]0, ∞[`. If not `None`, labels will
  potentially receive line breaks such that their bounding box is as close to
  the specified aspect ratio as possible. The aspect ratio is width/height.
  Default is `None`.
- `max_lines` is the maximum number of lines a label can be broken into when
  `target_aspect_ratio` is set. Default is `None` (no limit).
- `using` is an en existing `LabelPlacement` instance to use for labels. This
  allows reusing pre-computed label placements instead of calculating them
  from scratch.

**Returns:** either the label settings when all arguments are `Undefined` or `self`.

**Example:**

```python
scatter.label(by='group')

scatter.label(by=['state', 'city'], hierarchical=True)

scatter.label(by='city!', color='red', size=12)

scatter.label(by='category', exclude=['category:Other'])
```

### scatter.annotations(_annotations=Undefined_) {#scatter.annotations}

Set or get annotations.

**Arguments:**
- `annotations` is a list of annotations (`Line`, `HLine`, `VLine`, `Rect`, or `Contour`)

**Returns:** either the annotation properties when all arguments are `Undefined` or `self`.

**Example:**

```python
from jscatter import HLine, VLine
scatter.annotations([HLine(42), VLine(42)])
```

### scatter.tooltip(_enable=Undefined_, _properties=Undefined_, _histograms=Undefined_, _histograms_bins=Undefined_, _histograms_ranges=Undefined_, _histograms_size=Undefined_, _preview=Undefined_, _preview_type=Undefined_, _preview_text_lines=Undefined_, _preview_image_background_color=Undefined_, _preview_image_position=Undefined_, _preview_image_size=Undefined_, _preview_image_height=Undefined_, _preview_audio_length=Undefined_, _preview_audio_loop=Undefined_, _preview_audio_controls=Undefined_, _size=Undefined_) {#scatter.tooltip}

Set or get the tooltip settings.

**Arguments:**
- `enable` is a Boolean specifying if the tooltip should be enabled or disabled.

- `properties` is a list of string specifying for which visual or data properties to show in the tooltip. The visual properties can be some of `x`, `y`, `color`, `opacity`, and `size`. Note that visual properties are only shown if they are actually used to data properties. To reference other data properties, specify a column of the bound DataFrame by its name.

- `histograms` is a Boolean or list of property names specifying if histograms should be shown. When set to `True`, the tooltip will show histograms for all properties. Alternatively, you can provide a list of properties for which you want to show a histogram.

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

- `preview_image_height` The height of the image container pixels. By default, it is `None`, which makes the height deffault to 6em.

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


### scatter.camera(_target=Undefined_, _distance=Undefined_, _rotation=Undefined_, _view=Undefined_, _is_fixed=Undefined_) {#scatter.camera}

Get or set the camera view.

**Arguments:**
- `target` is a float tuple defining the view center.
- `distance` is a float value defining the distance of the camera from the scatter plot (imagine as a 2D plane in a 3D world).
- `rotation` is a float value defining the rotation in radians.
- `view` is an array-like list of 16 floats defining a view matrix.
- `is_fixed` is a Boolean value specifying whether the camera position is fixed to it's current location. If `True`, manual pan and zoom interactions are disabled. Note, you can still programmatically zoom via `scatter.zoom()`.

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


### scatter.lasso(_type=Undefined_, _color=Undefined_, _initiator=Undefined_, _min_delay=Undefined_, _min_dist=Undefined_, _on_long_press=Undefined_, _brush_size=Undefined_) {#scatter.lasso}

Get or set the lasso for selecting multiple points.

**Arguments:**
- `type` is a string specifying the lasso type. Must be one of `'freeform'`, `'brush'`, or `'rectangle'`.
- `color` is a string referring to a Matplotlib-compatible color.
- `initiator` is a Boolean value to specify if the click-based lasso initiator should be enabled or not.
- `min_delay` is an integer specifying the minimal delay in milliseconds before a new lasso point is stored. Higher values will result in more coarse grain lasso polygons but might be more performant. 
- `min_dist` is an integer specifying the minimal distance in pixels that the mouse has to move before a new lasso point is stored. Higher values will result in more coarse grain lasso polygons but might be more performant.
- `on_long_press` is a Boolean value specifying if the lasso should be activated upon a long press.
- `brush_size` is an integer specifying the size of the brush in pixels. This has only an effect if `type` is set to `'brush'`'. Defaults to `24`.

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

| Name                                     | Type <div style="min-width:250px"/>                                                                                                                                  | Default <div style="min-width:180px"/>     | Allow None | Read Only | Note <div style="min-width:320px"/> |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ---------- | --------- | ------------- |
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
| `tooltip_preview_image_height`           | int                                                                                                                                                                  | `None`                                     | `True`     |           |               |
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

- ![#56B4E9](https://placehold.co/16x16/56B4E9/56B4E9) `Sky blue (#56B4E9)`
- ![#E69F00](https://placehold.co/16x16/E69F00/E69F00) `Orange (#E69F00)`
- ![#009E73](https://placehold.co/16x16/009E73/009E73) `Bluish green (#009E73)`
- ![#F0E442](https://placehold.co/16x16/F0E442/F0E442) `Yellow (#F0E442)`
- ![#0072B2](https://placehold.co/16x16/0072B2/0072B2) `Blue (#0072B2)`
- ![#D55E00](https://placehold.co/16x16/D55E00/D55E00) `Vermillion (#D55E00)`
- ![#CC79A7](https://placehold.co/16x16/CC79A7/CC79A7) `Reddish Purple (#CC79A7)`
- ![#000000](https://placehold.co/16x16/000000/000000) `Black (#000000)`

**Example:**

```py
from jscatter import Scatter, okabe_ito
```

### glasbey_light {#glasbey-light}

A categorical color map consisting of 256 maximally distinct colors optimized for a _bright_ background. The colors were generated with the fantastic [Colorcet](https://colorcet.holoviz.org) package, which employs an algorithm developed by [Glasbey et al., 2007](https://strathprints.strath.ac.uk/30312/1/colorpaper_2006.pdf).

**Example:**

```py
from jscatter import Scatter, glasbey_light
```

### glasbey_dark {#glasbey-dark}

A categorical color map consisting of 256 maximally distinct colors optimized for a _dark_ background. The colors were generated with the fantastic [Colorcet](https://colorcet.holoviz.org) package, which employs an algorithm developed by [Glasbey et al., 2007](https://strathprints.strath.ac.uk/30312/1/colorpaper_2006.pdf).

**Example:**

```py
from jscatter import Scatter, glasbey_dark
```

### 2D Color Maps

We provide 2D colormaps from [pycolormap-2d](https://pypi.org/project/pycolormap-2d/).

- `ColorMap2DBremm`
- `ColorMap2DCubeDiagonal`
- `ColorMap2DSchumann`
- `ColorMap2DSteiger`
- `ColorMap2DTeuling2`
- `ColorMap2DZiegler`

**Example:**

```py
from jscatter import Scatter, ColorMap2DBremm
from matplotlib.colors import to_hex

cmap = ColorMap2DBremm(
    range_x=(df.x.min(), df.x.max()),
    range_y=(df.y.min(), df.y.max()),
)

group_cmap = {}

for group in df.groups.unique():
    mask = df.groups == group
    
    # Determine the median center of the group
    cx = df[mask].x.median()
    cy = df[mask].y.median()

    # Get color from Brenn's 2D color map
    color = cmap(cx, cy)

    # Convert to HEX
    group_cmap[group] = to_hex(color / 255)

scatter = Scatter(data=df, x='x', y='y', color_by='groups', color_map=group_cmap)
scatter.show()
```

### Utility Functions

Jupyter Scatter exposes the following utility functions for coloring points:

- `brighten(color: Color, factor: float)`
- `darken(color: Color, factor: float)`
- `saurate(color: Color, factor: float)`
- `desaturate(color: Color, factor: float)`

## Annotations

### HLine

A horizontal line annotation.

**Arguments:**

- `y` is a float value in the data space specifying the y coordinate at which the horizontal line should be drawn.
- `x_start` is a float value in the data space specifying the x coordinate at which the horizontal line should start. [optional]
- `x_end` is a float value in the data space specifying the x coordinate at which the horizontal line should end. [optional]
- `line_color` is a tuple of floats or string value specifying the line color. [optional]
- `line_width` is an Integer value specifying the line width. [optional]

**Examples:**

```python
from jscatter import plot, HLine
from numpy.random import rand
plot(
  x=rand(500),
  y=rand(500),
  annotations=[HLine(0)]
)
```

### VLine

A vertical line annotation.

**Arguments:**

- `x` is a float value in the data space specifying the x coordinate at which the vertical line should be drawn.
- `y_start` is a float value in the data space specifying the y coordinate at which the vertical line should start. [optional]
- `y_end` is a float value in the data space specifying the y coordinate at which the vertical line should end. [optional]
- `line_color` is a tuple of floats or string value specifying the line color. [optional]
- `line_width` is an Integer value specifying the line width. [optional]

**Examples:**

```python
from jscatter import plot, VLine
from numpy.random import rand
plot(
  x=rand(500),
  y=rand(500),
  annotations=[VLine(0)]
)
```

### Line

A line annotation.

**Arguments:**

- `vertices` is a list of float tuples in the data space specifying the line vertices.
- `line_color` is a tuple of floats or string value specifying the line color. [optional]
- `line_width` is an Integer value specifying the line width. [optional]

**Examples:**

```python
from jscatter import plot, Line
from numpy.random import rand
plot(
  x=rand(500),
  y=rand(500),
  annotations=[Line([(-1, -1), (0, 0), (1, 1)])]
)
```

### Rect

A rectangle annotation.

**Arguments:**

- `x_start` is a float value in the data space specifying the x coordinate at which the rectangle should start.
- `x_end` is a float value in the data space specifying the x coordinate at which the rectangle should end.
- `y_start` is a float value in the data space specifying the y coordinate at which the rectangle line should start.
- `y_end` is a float value in the data space specifying the y coordinate at which the rectangle line should end.
- `line_color` is a tuple of floats or string value specifying the line color. [optional]
- `line_width` is an Integer value specifying the line width. [optional]

**Examples:**

```python
from jscatter import plot, Rect
from numpy.random import rand
plot(
  x=rand(500),
  y=rand(500),
  annotations=[Rect(-1, 1, -1, 1)]
)
```

### Contour

A [contour line](https://en.wikipedia.org/wiki/Contour_line) annotation. Under
the hood the annotation uses [Seaborn's `kdeplot`](https://seaborn.pydata.org/generated/seaborn.kdeplot.html).

**Arguments:**

- `by` is a string value specifying a column of categorical values for generating separate contour lines. [optional]
- `line_color` is a tuple of floats or string value specifying the line color. [optional]
- `line_width` is an Integer value specifying the line width. [optional]
- `line_opacity_by_level` is a Boolean value specifying if the line opacity should be linearly increased from the lowest to the highest level such that the highest level is fully opaque. [optional]
- `kwargs` is a dictionary of additional arguments for [Seaborn's `kdeplot`](https://seaborn.pydata.org/generated/seaborn.kdeplot.html). [optional]

**Examples:**

```python
from jscatter import plot, Contour
from numpy.random import rand
plot(
  x=rand(500),
  y=rand(500),
  annotations=[Contour()]
)
```

## LabelPlacement

The `LabelPlacement` class handles the positioning of labels for data points
while managing collisions and optimizing label density using a tiling approach.

- [LabelPlacement](#LabelPlacement)
  - [Methods](#labelplacement.methods)
    - [compute()](#labelplacement.compute), [reset()](#labelplacement.reset), and [clone()](#labelplacement.clone)
    - [to_parquet()](#labelplacement.to_parquet) and [from_parquet()](#labelplacement.from_parquet)
    - [get_labels_from_tiles()](#labelplacement.get_labels_from_tiles)
  - [Properties](#labelplacement.properties)

### LabelPlacement(_data_, _by_, _x_, _y_, _\*\*kwargs_) {#LabelPlacement}

Creates a new label placement instance that positions labels based on data
coordinates while handling label collisions.

**Arguments:**

- `data` : pandas.DataFrame
  - DataFrame with x, y coordinates and categorical columns.
  
- `by` : str or list of str
  - Column name(s) used for labeling points. The referenced columns must contain either string or categorical values and are treated as categorical internally such that each category marks a group of points to be labeled as the category.
  
  - To display individual point labels (where each row gets its own label), append an exclamation mark to the column name. For instance, `"city!"` would indicate that each value in this column should be treated as a unique label rather than grouping identical values together.
  
  - Note: Currently only one column can be marked with an exclamation mark. If multiple columns are marked, only the first one is treated as containing point labels.

- `x` : str
  - Name of the x-coordinate column.
  
- `y` : str
  - Name of the y-coordinate column.
  
- `tile_size` : int, default=256
  - Size of the tiles used for label placement in pixels. This determines the granularity of label density control and affects how labels are displayed at different zoom levels.
  
- `importance` : str, optional
  - Column name containing importance values. These values determine which labels are prioritized when there are conflicts.
  
- `importance_aggregation` : {'min', 'mean', 'median', 'max', 'sum'}, default='mean'
  - Method used to aggregate importance values when multiple points share the same label. This affects how label importance is calculated for groups of points.
  
- `hierarchical` : bool, default=False
  - If True, the label types specified by `by` are expected to be hierarchical, which will affect the priority sorting of labels such that labels with a lower hierarchical index are displayed first.
  
- `zoom_range` : tuple of floats or list of tuple of floats or dict of tuple of floats, default=(-∞, ∞)
  - The range at which labels of a specific type (as specified with `by`) are allowed to appear. The zoom range is a tuple of zoom levels, where `zoom_scale == 2 ** zoom_level`. Default is (-∞, ∞) for all labels.
  
- `font` : Font or list of Font or dict of Font, default=DEFAULT_FONT_FACE
  - Font object(s) for text measurement. Can be specified as:
    - Single Font: Applied to all label types
    - List of Fonts: One font per label type in `by`
    - Dict: Maps label types or specific labels to fonts
  
- `size` : int or list of int or dict of int or 'auto', default='auto'
  - Font size(s) for label text. Can be specified as:
    - 'auto': Automatically assign sizes (hierarchical if hierarchical=True)
    - Single int: Uniform size for all labels
    - List of ints: One size per label type in `by`
    - Dict: Maps label types or specific labels to sizes
  
- `color` : color or list of color or dict of color or 'auto', default='auto'
  - Color specification for labels. Can be:
    - 'auto': Automatically choose based on background
    - str: Named color or hex code
    - tuple: RGB(A) values
    - list: Different colors for different hierarchy levels
    - dict: Mapping of label types or specific labels to colors
  
- `background` : color, default='white'
  - Background color. Used for determining label colors when color='auto'.
  
- `bbox_percentile_range` : tuple of float, default=(5, 95)
  - Range of percentiles to include when calculating the bounding box of points for label placement. This helps exclude outliers when determining where to place labels.
  
- `max_labels_per_tile` : int, default=100
  - Maximum number of labels per tile. Controls label density by limiting how many labels can appear in a given region. Set to 0 for unlimited.
  
- `scale_function` : {'asinh', 'constant'}, default='constant'
  - Label zoom scale function for zoom level calculations:
    - 'asinh': Scales labels by the inverse hyperbolic sine, initially increasing linearly but quickly plateauing
    - 'constant': No scaling with zoom, labels maintain the same size
  
- `positioning` : {'highest_density', 'center_of_mass', 'largest_cluster'}, default='highest_density'
  - Method used to determine the position of each label:
    - 'highest_density': Position label at the highest density point
    - 'center_of_mass': Position label at the center of mass of all points
    - 'largest_cluster': Position label at the center of the largest cluster
  
- `exclude` : list of str, default=[]
  - Specifies which labels should be excluded. Can contain:
    - Column names (e.g., `"country"`) to exclude an entire category
    - Column-value pairs (e.g., `"country:USA"`) to exclude specific labels
  
- `target_aspect_ratio` : float, optional
  - If not `None`, labels will potentially receive line breaks such that their bounding box is as close to the specified aspect ratio as possible. The aspect ratio is width/height.
  
- `max_lines` : int, optional
  - Specify the maximum number of lines a label should be broken into if `target_aspect_ratio` is not `None`.
  
- `verbosity` : {'debug', 'info', 'warning', 'error', 'critical'}, default='warning'
  - Controls the level of logging information displayed during label placement computation.

**Returns:** A new `LabelPlacement` instance.

**Examples:**

```python
from jscatter import LabelPlacement

# Basic usage
label_placement = LabelPlacement(
    data=df,
    by='country',
    x='longitude',
    y='latitude'
)

# Computing the labels
label_placement.compute()

# Use in a scatter plot
scatter.label(using=label_placement)
```

### Methods  {#labelplacement.methods}

#### compute(_show_progress=False_, _chunk_size=1024_) {#labelplacement.compute}

Compute the labels with full collision detection and density control.

**Arguments:**
- `show_progress` : bool, default=False
  - Whether to show a progress bar during computation.
- `chunk_size` : int, default=1024
  - The chunk size for parallel processing.

**Returns:**
- pandas.DataFrame - Computed labels ready for rendering

**Example:**
```python
label_placement = LabelPlacement(data=df, by='country', x='x', y='y')
labels = label_placement.compute(show_progress=True)
```

#### reset() {#labelplacement.reset}

Reset the computed labels, allowing spatial properties to be modified before recomputing labels.

**Note:** This method clears existing labels and tiles to allow spatial properties to be changed. Call compute() again after modifying properties.

**Example:**
```python
label_placement.reset()
label_placement.color = {'country': 'red'}
label_placement.compute()
```

#### clone(_\*\*kwargs_) {#labelplacement.clone}

Create a new LabelPlacement instance with the same configuration, optionally overriding specific parameters.

**Arguments:**
- `**kwargs` - Any parameters to override from the current instance

**Returns:**
- LabelPlacement - A new instance with the specified configuration

**Example:**
```python
# Clone the current instance but with different positioning
new_label_placement = label_placement.clone(positioning='center_of_mass')
```

#### to_parquet(_path_, _format='parquet'_) {#labelplacement.to_parquet}

Export label placement data to storage.

**Arguments:**
- `path` : str
  - Path where the data will be stored. For parquet format, this should be a directory.
  - For arrow_ipc format, this should be a file path.
- `format` : str, default="parquet"
  - Format to use for persistence. Options are "parquet" or "arrow_ipc".

**Example:**
```python
label_placement.to_parquet('my_dataset')
```

#### from_parquet(_path_, _format=None_) {#labelplacement.from_parquet}

Load label placement data from storage.

**Arguments:**
- `path` : str
  - Path where the data is stored. For parquet format, this should be a directory.
  - For "arrow_ipc" format, this should be a file path.
- `format` : "parquet" or "arrow_ipc", optional
  - Format to use for loading. Options are "parquet" or "arrow_ipc".
  - If None, will be determined from the path.

**Returns:**
- LabelPlacement - Loaded label placement object

**Example:**
```python
label_placement = LabelPlacement.from_parquet('my_dataset')
```

#### get_labels_from_tiles(_tile_ids_) {#labelplacement.get_labels_from_tiles}

Get labels from data tiles.

**Arguments:**
- `tile_ids` : list of str
  - Tile IDs

**Returns:**
- pandas.DataFrame - Labels from the data tiles

**Example:**
```python
tile_labels = labels.get_labels_from_tiles(['0,0,0', '1,0,0'])
```

### Properties {#labelplacement.properties}

#### computed {#labelplacement.computed}

Get whether labels have been computed or not.

**Returns:** 
- bool - If `True` the labels have been computed

#### loaded_from_persistence {#labelplacement.loaded_from_persistence}

Get whether this is a restored instance.

**Returns:** 
- bool - If `True` the labels have been restored from files.

#### background {#labelplacement.background}

Get or set the current background color.

**Returns:** 
- Color - The background color

#### color {#labelplacement.color}

Get or set the current font color mapping.

**Returns:** 
- Dict[str, str] - A dictionary mapping each label type and specific label to its color.

#### font {#labelplacement.font}

Get the current font face mapping.

**Returns:** 
- Dict[str, Font] - A dictionary mapping each label type and specific label to its font face.

**Note:** This property is read-only after labels have been computed as changing font faces may affect spatial placement.

#### size {#labelplacement.size}

Get the current font size mapping.

**Returns:** 
- Dict[str, int] - A dictionary mapping each label type and specific label to its size.

**Note:** This property is read-only after labels have been computed as changing font sizes may affect spatial placement.

#### zoom_range {#labelplacement.zoom_range}

Get the current zoom range mapping.

**Returns:** 
- Dict[str, Tuple[np.float64, np.float64]] - A dictionary mapping each label type and specific label to its zoom range.

**Note:** This property is read-only after labels have been computed as changing zoom ranges affects spatial placement.

#### exclude {#labelplacement.exclude}

Get or set the current exclude mapping.

**Returns:** 
- List[str] - A list of label types and specific labels to be excluded.

#### data {#labelplacement.data}

Get the input data.

**Returns:**
- pandas.DataFrame - The input data

#### x {#labelplacement.x}

Get the name of the x-coordinate column.

**Returns:**
- str - Column name

#### y {#labelplacement.y}

Get the name of the y-coordinate column.

**Returns:**
- str - Column name

#### by {#labelplacement.by}

Get the column name(s) defining the label hierarchy.

**Returns:**
- List[str] - Column names

#### hierarchical {#labelplacement.hierarchical}

Get whether the labels are hierarchical.

**Returns:**
- bool - True if labels are hierarchical

#### importance {#labelplacement.importance}

Get the name of the importance column.

**Returns:**
- Optional[str] - Column name or None

#### importance_aggregation {#labelplacement.importance_aggregation}

Get the importance aggregation method.

**Returns:**
- str - The aggregation method ('min', 'mean', 'median', 'max', or 'sum')

#### bbox_percentile_range {#labelplacement.bbox_percentile_range}

Get the percentile range for bounding box calculation.

**Returns:**
- Tuple[float, float] - The percentile range

#### tile_size {#labelplacement.tile_size}

Get the tile size.

**Returns:**
- int - Tile size in pixels

#### max_labels_per_tile {#labelplacement.max_labels_per_tile}

Get the maximum number of labels per tile.

**Returns:**
- int - Maximum number of labels per tile

#### scale_function {#labelplacement.scale_function}

Get the label zoom scale function.

**Returns:**
- str - The scale function ('asinh' or 'constant')

#### positioning {#labelplacement.positioning}

Get the label positioning method.

**Returns:**
- str - The positioning method ('highest_density', 'center_of_mass', or 'largest_cluster')

#### target_aspect_ratio {#labelplacement.target_aspect_ratio}

Get the target aspect ratio for line break optimization.

**Returns:**
- Optional[float] - The target aspect ratio or None

#### max_lines {#labelplacement.max_lines}

Get the maximum number of lines for line break optimization.

**Returns:**
- Optional[int] - The maximum number of lines or None

#### verbosity {#labelplacement.verbosity}

Get or set the current log level.

**Returns:**
- str - The current log level

#### labels {#labelplacement.labels}

Get the computed labels, if available.

**Returns:**
- Optional[pandas.DataFrame] - The computed labels or None

#### tiles {#labelplacement.tiles}

Get the tile mapping, if available.

**Returns:**
- Optional[Dict[str, List[int]]] - The tile mapping or None
