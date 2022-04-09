## v0.3.0

- **Breaking change:** Change the signature of `compose()` to simplify correspondence mapping of data points.
- Add ability to link the view of multiple scatter plots via `compose(sync_view=True)`
- Add `link()` as a shorthand for `compose(sync_view=True, sync_selection=True, sync_hover=True)`
- Rename `view_pixels` to `view_data` and add faster synchronization from the JS to Python kernel
- Fix two issues with the `order` argument in methods `color()`, `opacity()`, `size()`, `connection_color()`, `connection_opacity()`, and `connection_size()` that prevented it's propper use.
- Improve the ordering of the default Okabe Ito color map
- Expose default the Okabe Ito (`okabe_ito`) and Glasbey (`glasbey_light` and `glasbey_dark`) color maps for convenience

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
