# Traittypes

The source code in this folder is a fork from https://github.com/jupyter-widgets/traittypes.

The code contains the following two changes to the original code:
1. Fix: allow assignment of `None` to a DataFrame trait when `allow_none` is set to `True`.
2. Fix: remove `Empty` which relies on `Sentinel`, which is causing a `DeprecationWarning`. See https://github.com/jupyter-widgets/traittypes/issues/47
3. Fix: remove `dtype` from `DataFrame` due to `DeprecationWarning: metadata {'dtype': None} was set from the constructor. With traitlets 4.1, metadata should be set using the .tag() method`

We opted for forking the code as https://github.com/jupyter-widgets/traittypes seems unmaintained.
