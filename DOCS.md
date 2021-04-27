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
- `map` is either a string referencing a matplotlib color map, a matplotlib color map, or a list of colors.
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
- `map` is either a triple specifying an `np.linspace(*map)` or a list of opacities.
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
- `map` is either a triple specifying an `np.linspace(*map)` or a list of opacities.
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

Gets or sets the point connection

**Arguments:**

- `by` is either an array-like list of values or a string referencing a column in `data`.
- `order` is either a list of values (for categorical size encoding) or `reverse` to reverse the size map.
- `kwargs`:
  - `skip_widget_update` allows to skip the dynamic widget update when `True`. This can be useful when you want to animate the transition of multiple properties at once instead of animating one after the other.

**Returns:** either the x and y coordinate when x and y are `Undefined` or `self`.

**Examples:**

```python
scatter.size(by='price', map=(1, 0.25, 10))
```

## Properties

## Widget
