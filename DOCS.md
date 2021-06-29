# API Documentation

- [Constructors](#constructors)
- [Methods](#methods)
- [Properties](#properties)
- [Widget](#widget)

## Constructors

<a name="Scatter" href="#Scatter">#</a> <b>Scatter</b>(<i>x</i>, <i>y</i>, <i>data = None</i>, <i>\*\*kwargs</i>)

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

<a name="plot" href="#plot">#</a> <b>plot</b>(<i>x</i>, <i>y</i>, <i>data = None</i>, <i>\*\*kwargs</i>)

Short-hand function that creates a new scatter instance and immediately returns its widget.

**Arguments:** are the same as of [`Scatter`](#Scatter).

**Returns:** a new scatter widget.

**Examples:**

```python
from jscatter import plot
plot(x='speed', y='weight', data=cars, color='black', opacity_by='density', size=4)
```

## Methods

<a name="scatter.x" href="#scatter.x">#</a> scatter.<b>x</b>(<i>x = Undefined</i>, <i>\*\*kwargs</i>)

Gets or sets the x coordinate.

**Arguments:**

- `x` is either an array-like list of coordinates or a string referencing a column in `data`.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the x coordinate when x is `Undefined` or `self`.

**Examples:**

```python
scatter.x('price') # Triggers and animated transition of the x coordinates
```

<a name="scatter.y" href="#scatter.y">#</a> scatter.<b>y</b>(<i>y = Undefined</i>, <i>\*\*kwargs</i>)

Gets or sets the y coordinate.

**Arguments:**

- `y` is either an array-like list of coordinates or a string referencing a column in `data`.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the y coordinate when y is `Undefined` or `self`.

**Examples:**

```python
scatter.y('price') # Triggers and animated transition of the y coordinates
```

<a name="scatter.xy" href="#scatter.xy">#</a> scatter.<b>xy</b>(<i>x = Undefined</i>, <i>y = Undefined</i>, <i>\*\*kwargs</i>)

Gets or sets the x and y coordinate. This is just a convenience function to animate a change in the x and y coordinate at the same time.

**Arguments:**

- `x` is either an array-like list of coordinates or a string referencing a column in `data`.
- `y` is either an array-like list of coordinates or a string referencing a column in `data`.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the x and y coordinate when x and y are `Undefined` or `self`.

**Examples:**

```python
scatter.xy('size', 'speed') # Mirror plot along the diagonal
```

<a name="scatter.selection" href="#scatter.selection">#</a> scatter.<b>selection</b>(<i>selection = Undefined</i>)

Gets or sets the selected points.

**Arguments:**

- `selection` is either an array-like list of point indices.

**Returns:** either the x and y coordinate when x and y are `Undefined` or `self`.

**Examples:**

```python
scatter.selection(cars.query('speed < 50').index)
```

<a name="scatter.color" href="#scatter.color">#</a> scatter.<b>color</b>(<i>color = Undefined</i>, <i>color_active = Undefined</i>, <i>color_hover = Undefined</i>, <i>by = Undefined</i>, <i>map = Undefined</i>, <i>norm = Undefined</i>, <i>order = Undefined</i>, <i>\*\*kwargs</i>)

Gets or sets the point color.

**Arguments:**

- `color` is a valid matplotlib color.
- `color_active` is a valid matplotlib color.
- `color_hover` is a valid matplotlib color.
- `by` is either an array-like list of values or a string referencing a column in `data`.
- `map` is either a string referencing a matplotlib color map, a matplotlib color map object, a list of matplotlib-compatible colors, or `auto` (to let jscatter choose a default color map).
- `norm` is either a tuple defining a value range that's map to `[0, 1]` with `matplotlib.colors.Normalize` or a [matplotlib normalizer](https://matplotlib.org/stable/api/colors_api.html#classes).
- `order` is either a list of values (for categorical coloring) or `reverse` to reverse a color map.
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
  map=['red', 'green', 'blue'],
  order=['usa', 'europe', 'asia']
)
```

<a name="scatter.opacity" href="#scatter.opacity">#</a> scatter.<b>opacity</b>(<i>opacity = Undefined</i>, <i>by = Undefined</i>, <i>map = Undefined</i>, <i>norm = Undefined</i>, <i>order = Undefined</i>, <i>\*\*kwargs</i>)

Gets or sets the point opacity.

**Arguments:**

- `opacity` is a valid matplotlib color.
- `by` is either an array-like list of values, a string referencing a column in `data`, or `density`
- `map` is either a triple specifying an `np.linspace(*map)`, a list of opacities, or `auto` (to let jscatter choose a default opacity map).
- `norm` is either a tuple defining a value range that's map to `[0, 1]` with `matplotlib.colors.Normalize` or a [matplotlib normalizer](https://matplotlib.org/stable/api/colors_api.html#classes).
- `order` is either a list of values (for categorical opacity encoding) or `reverse` to reverse the opacity map.
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

<a name="scatter.size" href="#scatter.size">#</a> scatter.<b>size</b>(<i>size = Undefined</i>, <i>by = Undefined</i>, <i>map = Undefined</i>, <i>norm = Undefined</i>, <i>order = Undefined</i>, <i>\*\*kwargs</i>)

Gets or sets the point size.

**Arguments:**

- `size` is a valid matplotlib color.
- `by` is either an array-like list of values or a string referencing a column in `data`.
- `map` is either a triple specifying an `np.linspace(*map)`, a list of sizes, or `auto` (to let jscatter choose a default size map).
- `norm` is either a tuple defining a value range that's map to `[0, 1]` with `matplotlib.colors.Normalize` or a [matplotlib normalizer](https://matplotlib.org/stable/api/colors_api.html#classes).
- `order` is either a list of values (for categorical size encoding) or `reverse` to reverse the size map.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the x and y coordinate when x and y are `Undefined` or `self`.

**Examples:**

```python
scatter.size(by='price', map=(1, 0.25, 10))
```

<a name="scatter.connect" href="#scatter.connect">#</a> scatter.<b>connect</b>(<i>by = Undefined</i>, <i>order = Undefined</i>, <i>\*\*kwargs</i>)

Gets or sets the point connection.

**Description:** The `by` argument defines which points are part of a line segment. Points with the same value are considered to be part of a line segment. By default, points are connected in the order in which they appear the dataframe. You can customize that ordering via `order`.

**Arguments:**

- `by` is either an array-like list of integers or a string referencing a column in the dataframe.
- `order` is either a list of integers or a string referencing a column in the dataframe.
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

<a name="scatter.connection_color" href="#scatter.connection_color">#</a> scatter.<b>connection_color</b>(<i>color = Undefined</i>, <i>color_active = Undefined</i>, <i>color_hover = Undefined</i>, <i>by = Undefined</i>, <i>map = Undefined</i>, <i>norm = Undefined</i>, <i>order = Undefined</i>, <i>\*\*kwargs</i>)

Gets or sets the point connection color. This function behaves identical to [scatter.color()][scatter.color].

<a name="scatter.connection_opacity" href="#scatter.connection_opacity">#</a> scatter.<b>connection_opacity</b>(<i>opacity = Undefined</i>, <i>by = Undefined</i>, <i>map = Undefined</i>, <i>norm = Undefined</i>, <i>order = Undefined</i>, <i>\*\*kwargs</i>)

Gets or sets the point connection opacity. This function behaves identical to [scatter.opacity()][scatter.color].

<a name="scatter.connection_size" href="#scatter.connection_size">#</a> scatter.<b>connection_size</b>(<i>size = Undefined</i>, <i>by = Undefined</i>, <i>map = Undefined</i>, <i>norm = Undefined</i>, <i>order = Undefined</i>, <i>\*\*kwargs</i>)

Gets or sets the point connection size. This function behaves identical to [scatter.size()][scatter.color].

## Properties

## Widget
