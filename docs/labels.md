# Labels

When your points are labeled it can be helpful to show these labels in the
scatterplot. 

<div class="video">
  <video loop playsinline width="1280" data-name="labels">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/labels-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video and turn on audio</div>
</div>

You can draw text labels with Jupyter Scatter's `label()` function:

```py
scatter.label(by='df_column_name')
```

The referenced column needs to be either categorical or contain strings. For all
points with the same value, a text label is drawn.

For instance, say we have the following DataFrame:

|   | x | y | cat | pval |
|---|---|---|-----|------|
| 0 | 0 | 0 | A   | 0.51 |
| 1 | 1 | 0 | A   | 0.80 |
| 2 | 1 | 1 | A   | 0.25 |
| 3 | 0 | 1 | A   | 0.01 |
| 4 | 2 | 2 | B   | 0.65 |
| 5 | 3 | 2 | B   | 0.99 |
| 6 | 3 | 3 | B   | 1.33 |
| 7 | 2 | 3 | B   | 0.01 |

We can label points using the `cat` column via `scatter.label(by='cat')`.

<div class="img labels"><div /></div>

When displaying labels, Jupyter Scatter automatically manages label collision
and overcrowding. It uses an importance-based static placement strategy such
that overlapping labels with a lower priority are visualized at a higher zoom
when the collision is resolved. To handle many labels, Jupyter Scatter uses a
tiling approach where only a limited number of labels (default: `100`) are shown
per tile. 

You can control the label density with the `max_number` parameter. For instance,
to show fewer labels per tile do `scatter.label(by='cat', max_number=50)`.

::: info
For demo notebooks on how to use labels, see https://github.com/flekschas/jupyter-scatter-tutorial.
:::

## Importance

Whenever two text labels would collide, the label with the lower priority is
hidden. You can specify the importance via the `importance` parameter as shown
below. If no importance information is used, the number of points labeled by a
value is used.

```py
scatter.label(by='cat', importance='pval')
```

Since importance values are point specific, we need to aggregate multiple values
to derive the label importance. By default, Jupyter Scatter uses the mean but
you can change this behavior to `'min'`, `'median'`, `'max'`, or `'sum'`.

For instance, in the following we use the maximum point importance as the label
importance.

```py
scatter.label(by='cat', importance='pval', importance_aggregation='max')
```

Additionally, it's also possible to specify a custom aggregator function that
takes as input an array of floats and must return a single float.

::: tip
If you want to count the points and use this as the importance, simply omit
`importance` altogether as that's the default behavior.
:::

## Customization

### Appearance

You can customize the appearance and placement of labels in various ways. The
`font`, `color`, and `size` parameters allow you to adjust the font face, color,
and size.

```py{3-5}
scatter.label(
  by='cat',
  font='arial bold',
  color='red',
  size=36,
)
```

<div class="img labels-font"><div /></div>

By default, the label size is constant (i.e., zoom invariant) but you can also
enlarge labels as you soon in. To do this, set `scale_function` to `"asinh"`,
which enlarges the label using the inverse hyperbolic sine function.

```py{3-4}
scatter.label(
  by='cat',
  size=36,
  scale_function='asinh'
)
```

The inverse hyperbolic sine function is only applied when zooming in and
increases the label size sublinearly compared to the camera zoom as follows:

```
label_scale = asinh(zoom_scale) / asinh(1)
```

<div class="video">
  <video loop playsinline width="1474" data-name="labels-asinh-zoom-scale">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/labels-asinh-zoom-scale-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video and turn on audio</div>
</div>

::: info
Resolving collisions with inverse hyperbolic sine-scaled labels is computationally
more involved than constant scaling. Hence, if you have many labels (i.e., >=1000)
we recommend using constant scaling.
:::

### Position

You can also control the center position of labels, their alignment around
this position, and offset using the `positioning`, `align`, and `offset`
parameters.

```py{3-5}
scatter.label(
  by='cat',
  positioning='largest_cluster',
  align='top',
  offset=(2, 2),
)
```

<div class="img labels-positioning"><div /></div>

Jupyter Scatter offers three positioning algorithms with different tradeoffs as
outlined below. You may want to experiment to see which one works best for your
specific use case.

#### Highest Density
The default positioning method (`'highest_density'`) places the label at the
point of highest density within the group. This algorithm:

- Is fast
- Calculates density based on how many points are clustered in each area
- Places labels where the most points are concentrated
- Works well for irregular clusters with varying densities
- Gives okay results for many datasets

#### Center of Mass
The `'center_of_mass'` method places the label at the geometric center of all
points in the group:

- Is fast
- Calculates the center position using the Shoelace formula
- Creates a balanced label position
- Works well for single-cluster and evenly-distributed points
- Not recommended when your labels consist of disconnected clusters

#### Largest Cluster
The `'largest_cluster'` method identifies the largest sub-cluster within the
group using [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) and
places the label at its center of mass:

- Slowest method
- Detects clusters within each group
- Places the label at the center of mass of the largest cluster
- Works well when points naturally form multiple clusters
- Typically results in the best label placement overall

::: info
`"largest_cluster"` is an additional feature that's not included in the default
installation as it relies on HDBSCAN. To use the feature install Jupyter Scatter
via `pip install "jupyter-scatter[all]"`.
:::

### Line Breaks

If your labels tend to be on the longer side, you might want to
introduce line breaks. To make your life easier, you can specify an target
aspect ratio for which Jupyter Scatter will then try to find optimial line
breaks.

```py{3}
scatter.label(
  by='cat',
  target_aspect_ratio=5,
)
```

<div class="img labels-aspect-ratio"><div /></div>

## Point Labels

By default, all points with the same value in a column are grouped and given a
single label. However, sometimes you may want to label each individual data
point instead. Jupyter Scatter supports this through a special syntax - simply
append an exclamation mark to the column name:

For example, with the following DataFrame, using `scatter.label(by='city!')`
will label each individual city. Hence, even though there are two cities called
"Berlin", they each get their own label.

|   | x    | y    | city     |
|---|------|------|----------|
| 0 | 0.13 | 0.27 | Paris    |
| 1 | 0.87 | 0.93 | New York |
| 2 | 0.10 | 0.25 | Berlin   |
| 3 | 0.03 | 0.90 | Rome     |
| 4 | 0.19 | 0.78 | Tokyo    |
| 4 | 0.99 | 0.81 | Berlin   |

```py
scatter.label(by='city!')
```

<div class="img labels-point-labels"><div /></div>

This is useful when each data point represents a unique entity but their labels
are not unique (like cities on a map)

::: info
Currently, only one column can be marked as a point label. If multiple columns
are marked with exclamation marks, only the first one will be used as a point
label.
:::

## Multiple and Hierarchical Label Types

Jupyer Scatter also supports multiple and even hierarchical label types. For
instance, let's assume the data frame contains another categorical or string
column.

|   | x    | y    | cat | sub | pval |
|---|------|------|-----|-----|------|
| 0 | 0.13 | 0.27 | A   | A1  | 0.51 |
| 1 | 0.87 | 0.93 | B   | B1  | 0.80 |
| 2 | 0.10 | 0.25 | A   | A1  | 0.25 |
| 3 | 0.03 | 0.90 | A   | A2  | 0.01 |
| 4 | 0.19 | 0.78 | B   | B1  | 0.65 |

We can render out labels for both `"cat"` and `"sub"` as follows:


```py
scatter.label(by=["cat", "sub"])
```

<div class="img labels-multiple"><div /></div>

You can customize multiple labels in two ways. You can provide a list of values.
For instance, to draw `cat` labels in black bold at 24px and `sub` labels in
red italics at 18px, you can do:

```py{3-5}
scatter.label(
  by=['cat', 'sub']
  font=['bold', 'italic'],
  color=['black', '#ff0000'],
  size=[24, 18],
)
```

If you want to be even more specific with settings, you can also pass a
dictionary of `<type>:<value>` pairs. For instance, if you want all `cat` to be
black but `B` to be green, you can do the following:

```py{4}
scatter.label(
  by=['cat', 'sub'],
  font=['bold', 'italic'],
  color={'cat': 'black', 'cat:B': 'green', 'sub': '#ff0000'},
  size=[24, 18],
)
```

When working with multiple label types, collisions are still resolved in order
of importance such that the colliding labels with lower priorities appear only
at higher zoom levels when they no longer collide with labels of higher
importance. To adjust this behavior you can specify type-specific zoom ranges.
Zoom ranges are declared as zoom levels where `zoom_scale = 2 ^ zoom_level`.

```py{3}
scatter.label(
  by=['cat', 'sub'],
  zoom_ranges={'cat': (-math.inf, 2), 'sub': (2, 10)}
)
```

In the above example, `cat` labels are allowed to appear up until zoom level `2`
and `sub` labels are allowed to appear from zoom level `2` onward.

::: info
Note, zoom ranges do not enforce that labels are shown in that given range but
rather they specify the allowed zoom range at which they can appear given the
labels' importance and overlap with other labels.
:::

If your label types describe a strict hierarchy, as is the case for the example
specified above, then you can set `hierarchical` to `True`. For hierarchical
label types, Jupyter Scatter automatically enforces that labels in at a lower
hierarchical level are shown before labels with a higher hierarchical level.

```py{3}
scatter.label(
  by=['cat', 'sub'],
  hierarchical=True
)
```

For instance, in the example above, irrespective of the labels' importance,
`cat` labels will be shown before `sub` labels if they collide. Non-colliding
labels might be shown at the same time.

## Exclude Labels

Sometimes you do not want to show all labels. For instance, when clustering
labels it's common to label unclear points as _noise_. You can exclude unwanted
labels using the `exclude` parameter. For instance, in the following we exclude
the label `B`.

```py
scatter.label(by=['cat'], exclude=['B'])
```

When using multiple label types, you need to specify excluded labels via
`<type>:<label>`. For instance, in the following we exclude `sub` label `A2`

```py
scatter.label(by=['cat', 'sub'], exclude=['sub:A2'])
```

## Precompute Labels

When working with many labels it can take a moment to compute the labels. If you
want to use the same labels in multiple scenarios, it can be beneficial to
precompute labels and later load them from file.

To precompute labels, use the `LabelPlacement` class. It accepts many of the
same parameters as the `Scatter`'s `label` function.

```py
labels = LabelPlacement(data=df, x='x', y='y', by='cat')
labels.compute()
```

::: tip
For tracking the progress while precomputing very large label sets you can show
a progress bar via `labels.compute(show_progress=True)`. This feature requires
the complete installation via `pip install "jupyter-scatter[all]"`.
:::

Once the labels placement has been precomputed, you can persist them to disk as
parquet files.

```py
labels.to_parquet('my_labels')
```

Later on you can then recreate the label placement instance.

```py
labels.from_parquet('my_labels')
```

Importantly, you can pass this label placement instance directly to your scatter
instance.

```py
scatter.labels(using=labels)
```

::: info
The label placement class is responsible for statically resolving label
collisions by determining at which zoom level labels should appear. This
calculation is based on a tile size parameter, which defaults to 256 × 256
pixels.

It's important to understand that font size and zoom ranges are relative to this
tile size. When using `Scatter`'s `label()` function directly, the tile size
defaults to height × height (the widget's height), which means labels are
optimized for the initial view.

When precomputing labels with a different tile size than your widget's height,
you may notice differences in how labels appear:

- If you specify a smaller tile size (e.g., `100`) for a taller visualization
  (e.g., `height=200`), labels using the `'asinh'` scale function will appear
  larger than expected in the initial view.
- With the `'constant'` scale function, font size remains consistent regardless
  of tile size.

For most reliable results when precomputing labels for various display sizes,
the `'constant'` scale function is recommended.
:::

<style scoped>
  .img {
    max-width: 100%;
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
  }

  .img.labels {
    width: 842px;
    background-image: url(https://storage.googleapis.com/jupyter-scatter/dev/images/labels-light.png)
  }
  .img.labels div { padding-top: 56.29453682% }
  :root.dark .img.labels {
    background-image: url(https://storage.googleapis.com/jupyter-scatter/dev/images/labels-dark.png)
  }

  .img.labels-font {
    width: 848px;
    background-image: url(https://storage.googleapis.com/jupyter-scatter/dev/images/labels-font-light.png)
  }
  .img.labels-font div { padding-top: 55.89622642% }
  :root.dark .img.labels-font {
    background-image: url(https://storage.googleapis.com/jupyter-scatter/dev/images/labels-font-dark.png)
  }

  .img.labels-positioning {
    width: 1919px;
    background-image: url(https://storage.googleapis.com/jupyter-scatter/dev/images/labels-positioning-light.png)
  }
  .img.labels-positioning div { padding-top: 31.57894737% }
  :root.dark .img.labels-positioning {
    background-image: url(https://storage.googleapis.com/jupyter-scatter/dev/images/labels-positioning-dark.png)
  }

  .img.labels-aspect-ratio {
    width: 846px;
    background-image: url(https://storage.googleapis.com/jupyter-scatter/dev/images/labels-aspect-ratio-light.png)
  }
  .img.labels-aspect-ratio div { padding-top: 57.21040189% }
  :root.dark .img.labels-aspect-ratio {
    background-image: url(https://storage.googleapis.com/jupyter-scatter/dev/images/labels-aspect-ratio-dark.png)
  }

  .img.labels-point-labels {
    width: 848px;
    background-image: url(https://storage.googleapis.com/jupyter-scatter/dev/images/labels-point-labels-light.png)
  }
  .img.labels-point-labels div { padding-top: 55.66037736% }
  :root.dark .img.labels-point-labels {
    background-image: url(https://storage.googleapis.com/jupyter-scatter/dev/images/labels-point-labels-dark.png)
  }

  .img.labels-multiple {
    width: 960px;
    background-image: url(https://storage.googleapis.com/jupyter-scatter/dev/images/labels-multiple-light.png)
  }
  .img.labels-multiple div { padding-top: 49.58333333% }
  :root.dark .img.labels-multiple {
    background-image: url(https://storage.googleapis.com/jupyter-scatter/dev/images/labels-multiple-dark.png)
  }

  .video {
    position: relative;
  }

  .video video {
    filter: blur(0.5px);
  }

  .video .overlay {
    position: absolute;
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    user-select: none;
    pointer-events: none;
    transition: 0.25s ease;
    border-radius: 0.25rem;
    font-weight: 700;
    color: black;
    background: rgba(255, 255, 255, 0.5);
  }

  .video:hover .overlay {
    opacity: 0;
  }

  .video:hover video {
    filter: blur(0);
  }

  :root.dark .video .overlay {
    color: white;
    background: rgba(30, 30, 32, 0.5);
  }
</style>


<script setup>
  import { videoColorModeSrcSwitcher, videoPlayOnHover } from './utils';
  videoColorModeSrcSwitcher();
  videoPlayOnHover();
</script>
