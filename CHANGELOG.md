## v0.7.1

- Take the x-padding into account when setting a fixed width ([#36](https://github.com/flekschas/jupyter-scatter/issues/36))
- Make `width` and `height` correspond to the canvas' (i.e., inner) dimensions such that `Scatter(width=500, height=500)` will lead to a canvas of size 500x500 px.

## v0.7.0

- Add support for legends via `scatter.legend(True, position='top-right', size='small')` ([#30](https://github.com/flekschas/jupyter-scatter/issues/30))

## v0.6.1

- Remove accidentally added `console.log`

## v0.6.0

- Add support for axes labels via `scatter.axes(labels=True)` or `scatter.axes(labels=['x-axis', 'y-axis'])` ([#29](https://github.com/flekschas/jupyter-scatter/issues/29))

## v0.5.1

- Fix issues when specifying the color, opacity, or size map via a `dict`

## v0.5.0

- **Breaking changes:**
  - For `scatter.color()`, rename `color` to `default`, `color_selected` to `selected`, and `color_hover` and `hover`
  - For `scatter.opacity()`, rename `opacity` to `default`
  - For `scatter.size()`, rename `size` to `default`
  - For `scatter.connection_color()`, rename `color` to `default`, `color_selected` to `selected`, and `color_hover` and `hover`
  - For `scatter.connection_opacity()`, rename `opacity` to `default`
  - For `scatter.connection_size()`, rename `size` to `default`
- Add `scatter.opacity(unselected=0.5)`. This property defines the opacity scaling factor of unselected points and must be in `[0, 1]`. This scaling is only applied if one or more points are selected.

## v0.4.1

- Fix an issue when dynamically resizing the scatter plot height
- Fix an issue when switching from categorical to continuous color/opacity/size encoding

## v0.4.0

- **Breaking change:** Renamed `color_active` and `connection_color_active` to `color_selected` and `connection_color_selected` respectively for clarity.
- Add support for axes via `Scatter(axes=True, axes_grid=True)` or `scatter.axes(True, grid=True)`.
- Add support for log and power x/y scales via `Scatter(x_scale='log')` or `scatter.y(scale='pow')`.
- Add docstrings and type hints

## v0.3.4

- Unset data-driven color, opacity, and size encoding when only a constant value was passed to an encoder function. E.g., after initializing `scatter = Scatter(..., color_by='property')`, calling `scatter.color('red')` will automatically unset `color_by`.

## v0.3.3

- Fix `scatter.pixels()`

## v0.3.2

- Fix more type hints...

## v0.3.1

- Make type hints backward compatible to Python 3.7

## v0.3.0

- **Breaking change:** Change the signature of `compose()` to simplify correspondence mapping of data points.
- **Breaking change:** Rename `view_pixels` to `view_data` and add faster synchronization from the JS to Python kernel
- Add ability to link the view of multiple scatter plots via `compose(sync_view=True)`
- Add `link()` as a shorthand for `compose(sync_view=True, sync_selection=True, sync_hover=True)`
- Add ability to defined categorical colors using a dictionary. E.g., `scatter.color(by='coolness', map=dict(cool='blue', hot='orange'))`
- Fix two issues with the `order` argument in methods `color()`, `opacity()`, `size()`, `connection_color()`, `connection_opacity()`, and `connection_size()` that prevented it's propper use.
- Improve the ordering of the default Okabe Ito color map
- Expose default the Okabe Ito (`okabe_ito`) and Glasbey (`glasbey_light` and `glasbey_dark`) color maps for convenience
- Automatiecally handle string as categorical data

## v0.2.2

- Fix broken installation of [v0.2.1](v0.2.1) (#23)

## v0.2.1

- Simplify installation ([#16](https://github.com/flekschas/jupyter-scatter/pull/16))

## v0.2.0

- Complete rewrite of the Python API to match ([#3](https://github.com/flekschas/jupyter-scatter/issues/3))

## v0.1.2

- Properly destroy regl-scatterplot on destroy of a widget instance

## v0.1.1

- Fix bunch of typos related to renaming the pypi and npm package

## v0.1.0

- First version
