## v0.18.1

- Fix: re-enable point transition and make it explicit via `scatter.options(transition_points=True, transition_points_duration=3000)`
- Fix: Correctly render axes grid lines upon resizing
- Fix: Reapply point color map when setting `color_by`
- Fix: Do not set tooltip titles if the corresponding elements are undefined.

## v0.18.0

- Feat: add support for line-based annotations via `scatter.annotations()`
- Feat: the exported and saved images now include a background color instead of a transparent background. You can still still export or save images with a transparent background by holding down the ALT key while clicking on the download or camera button.
- Refactor: When saving the current view as an image via the camera button on the left side bar, the image gets saved in `scatter.widget.view_data` as a 3D Numpy array (shape: `[height, width, 4]`) instead of a 1D Numpy array. Since the shape is now encoded by the 3D numpy array, `scatter.widget.view_shape` is no longer needed and is removed.
- Fix: hide button for activating rotate mouse mode as the rotation did not work (which is easily fixable) and should not be available when axes are shown as the axes are not rotateable. However rotating the plot without rotating the axis leads to incorrect tick marks.
- Fix: VSCode integration by updating regl-scatterplot to v1.10.4 ([#37](https://github.com/flekschas/jupyter-scatter/issues/37))

## v0.17.2

- Fix: bump regl-scatterplot to v1.10.2 for a [hotfix related to rendering more than 1M points](https://github.com/flekschas/regl-scatterplot/pull/190)

## v0.17.1

- Fix: regression preventing tooltip from showing up ([#141](https://github.com/flekschas/jupyter-scatter/issues/141))

## v0.17.0

- Feat: add `scatter.show_tooltip(point_idx)`
- Fix: reset scale & norm ranges upon updating the data via `scatter.data()`
- Fix: ensure `scatter.axes(labels=['x_label', 'y_label'])` works properly ([#137](https://github.com/flekschas/jupyter-scatter/issues/137]))

## v0.16.1

- Fix: preserve filter state upon changing visual encoding ([#134](https://github.com/flekschas/jupyter-scatter/issues/134))

## v0.16.0

**BREAKING CHANGES**:

The following list of helper widgets for configuring the scatter are removed from the `scatter.widget` as they are unmaintained, undocumented, and incomplete. If you relied on any of those UI widgets, please see `jscatter/widget.py` v0.15.0 on how they were created.

- `scatter.widget.mouse_mode_widget`
- `scatter.widget.lasso_initiator_widget`
- `scatter.widget.selection_widget`
- `scatter.widget.hovering_widget`
- `scatter.widget.color_widgets`
- `scatter.widget.color_by_widget`
- `scatter.widget.color_map_widget`
- `scatter.widget.height_widget`
- `scatter.widget.background_color_widget`
- `scatter.widget.background_image_widget`
- `scatter.widget.lasso_color_widget`
- `scatter.widget.lasso_min_delay_widget`
- `scatter.widget.lasso_min_dist_widget`
- `scatter.widget.color_widget`
- `scatter.widget.color_selected_widget`
- `scatter.widget.color_hover_widget`
- `scatter.widget.opacity_widget`
- `scatter.widget.selection_outline_width_widget`
- `scatter.widget.size_widget`
- `scatter.widget.selection_size_addition_widget`
- `scatter.widget.reticle_widget`
- `scatter.widget.reticle_color_widget`
- `scatter.widget.download_view_widget`
- `scatter.widget.save_view_widget`
- `scatter.widget.reset_view_widget`

Additionally, the following helper methods are removed as they are unnecessary.

- `scatter.widget.options()` (simply listed out all above removed widgets)
- `scatter.widget.select()` (same as `scatter.selection = list_of_point_indices`)
- `scatter.widget.use_cmap()` (same as passing the cmap name to `scatter.color(map=cmap_name)`)

**Other Changes**:

- Feat: Add basic support for x/y time scale via `Scatter(data=df, x='x', x_scale='time', y='y', y_scale='time')`
- Docs: Add API documentation for `scatter.widget`
- Docs: Add description for x/y scales
- Docs: Add description for connected scatterplots
- Fix: Match numerical and string IDs properly in `compose(match_by='XYZ')`
- Fix: Ensure that the domain and histograms match by avoiding missing categorical indices
- Fix: Ignore `NaN`s when computing histograms
- Fix: Warn when data contains `NaN`s and replace them with zeros
- Fix: Show correctly ordered color encoding in legend
- Fix: Ensure the widget's x and y scale domains are updated properly
- Fix: Ensure the widget's color, opacity, and size titles are updated properly
- Fix: Ensure the widget's axes titles are updated properly
- Fix: Include normalization in data dimension name
- Fix: Allow rendering a single axis instead of enforcing either none or both axis
- Fix: Rely on pre-normalized data to get bin ID
- Fix: Connect order
- Fix: X/Y scale domain bug
- Fix: Connected point bugs

## v0.15.1

- Fix: Remove an unused widget property that causes an issue with newer version of anywidget ([#117](https://github.com/flekschas/jupyter-scatter/pull/117))

## v0.15.0

- Feat: Add support for histograms in the tooltip ([#96](https://github.com/flekschas/jupyter-scatter/pull/96))
- Feat: Add support for non-visualized properties in the tooltip ([#96](https://github.com/flekschas/jupyter-scatter/pull/96))
- Fix: Allow mixing custom and DataFrame-based data ([#89](https://github.com/flekschas/jupyter-scatter/issues/89))
- Fix: Improve the tooltip positioning to avoid the tooltip being cut off unnecessarily
- Fix: Properly redraw axes on resize ([#108](https://github.com/flekschas/jupyter-scatter/issues/108))
- Fix: Incorrect axes scale domain ([#107](https://github.com/flekschas/jupyter-scatter/issues/107))
- Fix: Use custom regl-scatterplot option on creating a `Scatter` instance ([#106](https://github.com/flekschas/jupyter-scatter/issues/106))
- Fix: Broken link to properties in docstrings ([#110](https://github.com/flekschas/jupyter-scatter/issues/110))

## v0.14.3

- Fix: don't return color, opacity, or size settings when `labeling` is defined
- Fix: prevent x-axis label to be cut off at the bottom
- Fix: axes label color in dark mode

## v0.14.2

- Fix view synchronization when axes are _not_ shown
- Fix y-padding size determination
- Fix stale channel value getter for the tooltip

## v0.14.1

- Fix: update `color`, `opacity`, and `size` scales as the domains update
- Fix: auto-reset `x` and `y` scale domains upon updating the `x` and `y` data
- Fix: use better number formatter for the legend

## v0.14.0

- Add the ability to show a tooltip upon hovering over a point via `scatter.tooltip(true)` ([#86](https://github.com/flekschas/jupyter-scatter/pull/86))
- Fix axes updating of linked scatter plots when panning and zooming ([#87](https://github.com/flekschas/jupyter-scatter/issues/87))
- Fix missing x-axes of linked scatter plots ([#84](https://github.com/flekschas/jupyter-scatter/issues/84))
- Fix a type in the return value of `scatter.xy()`

## v0.13.1

- Fix: Prevent resetting the filter upon color, size, or opacity changes
- Fix: Upon changing the associated data frame via `scatter.data(new_df)`, reapply color, size, and opacity settings

## v0.13.0

- Add ability to specify titles when composing multiple scatter plots by passing tuples of `(Scatter, title)` to `compose()` or `link()`
- Fix: Add docstrings to `compose()` and `link()`
- Fix: Optimize height of the legend
- Fix: Check if axes are enabled before updating them when the x or y scale changes
- Fix: Merge point selections on `SHIFT` instead of activating the lasso as `SHIFT` interferes with Jupyter Lab
- Fix: Allow to call `scatter.zoomTo()` with the same points multiple times
- Fix: Unfilter when calling `scatter.filter(None)`
- Fix: Properly listen to changes when setting custom `regl-scatterplot` options via `scatter.options()`

## v0.12.6

- Fix distributed build by ensuring that `jscatter/bundle.js` is included in the build
- Fix categorical encoding for partial data

## v0.12.5

> **Warning**: do not use this version! The distributed build is broken. Use `v0.12.6` instead. :pray:

- Ensure that the default point colors respect the background when setting both at the same time during initialization. I.e., in the following scenario, the point color will be set to _white_ by default as the background color was set to _black_:

  ```py
  jscatter.plot(data=df, background_color='black')
  ```

- Fix an issue when working with views of a pandas DataFrame where not all categorical data is present

- Loosen strictness of `rows` and `cols` of `compose()` to allow having empty cells in the grid
  

## v0.12.4

- Respect the dictionary key-value order of categorical encoding maps in the legend. E.g., the following categorical color map legend will read `C`, then `B`, and finally `A`:

  ```py
  scatter.legend(True)
  scatter.color(map=dict(C='red', B='blue', A='pink'))
  ```

- Update third-party JS libraries

## v0.12.3

- Fix incorrect legend for categorical coloring

## v0.12.2

- Update `regl-scatterplot` to `v1.6.9`

## v0.12.1

- Fix the ordering of the legend's value labels for continuous encodings such that high to low values are order top to bottom ([#70](https://github.com/flekschas/jupyter-scatter/issues/70))

## v0.12.0

- Add support for referencing points by the Pandas DataFrame's index via `Scatter(data_use_index=True)` or `scatter.data(use_index=True)`. This is useful for synchronizing the selection or filtering of two Scatter instances that operate on different data frames with point correspondences. ([#62](https://github.com/flekschas/jupyter-scatter/issues/62)

  ```py
  import jscatter
  import numpy as np
  import pandas as pd

  df1 = pd.DataFrame(
      data=np.random.rand(16, 2),
      index=[chr(65 + x) for x in range(16)],
      columns=['x', 'y']
  )
  df2 = pd.DataFrame(
      data=np.random.rand(8, 2),
      index=[chr(76 - x) for x in range(8)],
      columns=['x', 'y']
  )

  s1 = jscatter.Scatter(data=df1, data_use_index=True, x='x', y='y', x_scale=[0, 2], y_scale=[0, 1])
  s2 = jscatter.Scatter(data=df2, data_use_index=True, x='x', y='y', x_scale=[-1, 1], y_scale=[0, 1])

  def on_selection_change(change):
      s2.selection(s1.selection())

  s1.widget.observe(on_selection_change, names='selection')

  jscatter.compose([s1, s2])
  ```

  https://user-images.githubusercontent.com/932103/223899982-d2837c4d-f486-4f33-af22-cf3866c4983e.mp4

- Avoid unregistering all observers when calling `jscatter.compose()` such that external observers remain registered
- Fix undefined `this` in codec preventing the `scatter.selection()` from working correctly ([#66](https://github.com/flekschas/jupyter-scatter/pull/66))

## v0.11.0

- Add `scatter.filter([0, 1, 2])` for filtering down points. Filtering down to a specific subset of points is much faster than than updating the underlying data ([#61](https://github.com/flekschas/jupyter-scatter/issues/61))
- Add `scatter.data(df)` to allow rendering new data points without having to re-initialize the scatter instance ([#61](https://github.com/flekschas/jupyter-scatter/issues/61))
- Add the ability to automatically zoom to filtered points via `Scatter(zoom_on_filter=True)` or `scatter.zoom(on_filter=True)`
- Add lasso on long press and make it the default. The behavior can be changed via `Scatter(lasso_on_long_press=False)` or `scatter.lasso(on_long_press=False)`
- Updated `regl-scatterplot` to `v1.6`

## v0.10.0

- Add support for automatic zooming to selected points via `scatter.zoom(on_selection=True)`
- Fix view synchronization issue
- Add `remove()` to the JS widget to ensure that the scatterplot is destroyed in ipywidgets `v8`.

## v0.9.0

- Add support for animated zooming to a set of points via `scatter.zoom(pointIndices)` ([#49](https://github.com/flekschas/jupyter-scatter/issues/49))
- Bump regl-scatterplot to `v1.4.1`
- Add support for VSCode and Colab ([#37](https://github.com/flekschas/jupyter-scatter/issues/37))
- Fix serving of numpy data for JS client. Use consistent serialization object between JS and Python.

## v0.8.0

- Add support for end labeling of continuous data encoding via a new `labeling` argument of `color()`, `opacity()`, `size()`, `connection_color()`, `connection_opacity()`, and `connection_size()`. ([#46](https://github.com/flekschas/jupyter-scatter/pull/46))
- Fix the incorrect size legend when the size map is reversed ([#47](https://github.com/flekschas/jupyter-scatter/issues/47))

## v0.7.4

- Adjust widget to be compatible with ipywidgets `v8` and jupyterlab-widgets `v3` ([#39]((https://github.com/flekschas/jupyter-scatter/issues/39)))

## v0.7.3

- Fix broken build of `v0.7.2`
- Fix versions of ipywidgets and jupyterlab_widgets to avoid running into incompatibilities ([#40](https://github.com/flekschas/jupyter-scatter/issues/40))

## v0.7.2

**Do not use. Build is broken**

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
