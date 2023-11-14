# Axes, Legend, & Tooltip

Now that we know how to create, configure, compose, and link scatter plots, it's
time learn axes, legends, and tooltip which greatly help in making sense of the
visualized data.

## Axes

You might have noticed already that axes are drawn by default. E.g.:

```py{7}
from jscatter import Scatter
from numpy.random import rand

scatter = jscatter.Scatter(x=rand(500), y=rand(500))
scatter.show()
```

<div class="img axes"><div /></div>

::: info
You can also hide the axes via `scatter.axes(False)` in case they are not
informative like for t-SNE or UMAP embeddings.
:::

In addition, you can also enable a grid, which can be helpful to better locate
points.

```py
scatter.axes(grid=True)
```

<div class="img axes-grid"><div /></div>

And finally, you can also label the axes

```py
scatter.axes(labels=['Speed (km/h)', 'Weight (tons)'])
```

<div class="img axes-labels"><div /></div>

## Legend

When you encode data properties with the point color, opacity, or size, it's
immensly helpful to know how the encoded data properties relate to the visual
properties by showing a legend.

```py{22-24}
import jscatter
import numpy as np
import pandas as pd

df = pd.DataFrame({
  # Random floats
  "mass": np.random.rand(500),
  "speed": np.random.rand(500),
  "pval": np.random.rand(500),
  # Gaussian-distributed floats
  "effect_size": np.random.normal(.5, .2, 500),
  # Random letters A, B, C, D, E, F, G, H
  "cat": np.vectorize(lambda x: chr(65 + round(x * 8)))(np.random.rand(500)),
  # Random letters X, Y, Z
  "group": np.vectorize(lambda x: chr(88 + round(x * 2)))(np.random.rand(500)),
})

scatter = jscatter.Scatter(
  data=df,
  x="mass",
  y="speed",
  color_by="cat",
  size_by="pval",
  legend=True,
)
scatter.show()
```

<div class="img legend-1"><div /></div>

When you encode a categorical data property (like `cat`) using color, Jupyter
Scatter will list out each category in the legend. In contrast, for continuous
data properties (like `pval`), only five values are shown in the legend: the
minimum, maximum, and three equally spaced values in between.

```py
scatter.color(by="pval").opacity(by="cat").size(5)
```

<div class="img legend-2"><div /></div>

Notice how the legend now only shows five entries for `color` as it encodes a
continuous variable.

In addition to just showing a mapping of data and visual properties, Jupyter
Scatter can also label continuous properties.

```py
scatter.color(labeling={
    "variable": "p-value",
    "minValue": "significant",
    "maxValue": "insignificant", 
})
```

<div class="img legend-3"><div /></div>

## Tooltip

Legends depict how data are mapped to visual properties, yet require repeated
eye movement between individual points and the legend for accurate
interpretation. Jupyter Scatter supports a tooltip to show a point's encoded
properties and related details, alleviating this strain.

```py
scatter.tooltip(True)
```

<div class="img tooltip-1"><div /></div>

Each row in the tooltip corresponds to a property. From left to right, each
property features the:

1. visual channel and property like `x`, `y`, `color`, `opacity`, or `size` (if the property is for visual encoding)
2. name as specified by the column name in the bound DataFrame
3. actual data value
4. histogram or treemap of the data property's distribution

<div class="img tooltip-2"><div /></div>

For numerical properties, the histogram is visualized as a bar chart. For
categorical properties, the histogram is visualized as a
flat [treemap](https://en.wikipedia.org/wiki/Treemapping) where the rectangles
represents the proportion of categories compared to the whole. Treemaps are
useful in scenarios with a lot of categories as shown below.

<div class="img tooltip-treemap"><div /></div>

In both cases, the highlighted bar indicates how the hovered point compares to
the other points.

By default, the tooltip shows all properties that are visually encoded but you
can limit the contents of the tooltip as follows:

```py
scatter.tooltip(contents=["color", "opacity"])
```

<div class="img tooltip-3"><div /></div>

Importantly, you can also show other properties in the tooltip that are not
directly visualized with the scatter plot. Other properties have to be
referenced by their respective column names.

```py{5-6}
scatter.tooltip(
  contents=[
    "color",
    "opacity",
    "group",
    "effect_size",
  ]
)
```

<div class="img tooltip-4"><div /></div>

Here, for instance, we're showing the point's `group` and `effect_size`
properties, which are two other DataFrame columns we didn't visualize.

::: tip
The order of `contents` defines the order of the tooltip entries.
:::

### Customizing Numerical Histograms

The histograms of numerical data properties consists of `20` bins, by default,
and is covering the entire data range, i.e., it starts at the minumum and ends
at the maximum value. You can adjust both aspects either globally for all
histograms as follows:

```py
scatter.tooltip(histograms_bins=40, histograms_ranges=(0, 1))
```

<div class="img tooltip-5"><div /></div>

To customize the number of bins and the range by content you can do:

```py
scatter.tooltip(
  histograms_bins={"color": 10, "effect_size": 30},
  histograms_ranges={"color": (0, 1), "effect_size": (0.25, 0.75)}
)
```

<div class="img tooltip-6"><div /></div>

Since an increased number of bins can make it harder to read the histogram, you
can adjust the size as follows:

```py
scatter.tooltip(histograms_size="large")
```

<div class="img tooltip-7"><div /></div>

If you set the histogram range to be smaller than the data extent, some points
might lie outside the histogram. For instance, previously we restricted the
`effect_size` to `[0.25, 0.75]`, meaning we disregarded part of the lower and
upper end of the data.

In this case, hovering a point with an `effect_size` less than `.5` will be
visualized by a red `]` to the left of the histogram to indicate it's value is
smaller than the value represented by the left-most bar.

<div class="img tooltip-8"><div /></div>

Likewise, hovering a point with an `effect_size` larger than `0.75` will be
visualized by a red `[` to the right of the histogram to indicate it's value is
larger than the value represented by the right-most bar.

<div class="img tooltip-9"><div /></div>

Finally, if you want to transform the histogram in some other way, use your
favorite method and save the transformed data before referencing it. For
instance, in the following, we winsorized the `effect_size` to the `[10, 90]`
percentile:

```py
from scipy.stats.mstats import winsorize

df['effect_size_winsorized'] = winsorize(df.effect_size, limits=[0.1, 0.1])
scatter.tooltip(contents=['effect_size_winsorized'])
```

<div class="img tooltip-10"><div /></div>

<style scoped>
  .img {
    max-width: 100%;
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
  }

  .img.axes {
    width: 596px;
    background-image: url(/images/axes-light.png)
  }
  .img.axes div { padding-top: 48.489933% }

  :root.dark .img.axes {
    background-image: url(/images/axes-dark.png)
  }

  .img.axes-grid {
    width: 597px;
    background-image: url(/images/axes-grid-light.png)
  }
  .img.axes-grid div { padding-top: 47.906198% }

  :root.dark .img.axes-grid {
    background-image: url(/images/axes-grid-dark.png)
  }

  .img.axes-labels {
    width: 597px;
    background-image: url(/images/axes-labels-light.png)
  }
  .img.axes-labels div { padding-top: 50.921273% }

  :root.dark .img.axes-labels {
    background-image: url(/images/axes-labels-dark.png)
  }

  .img.legend-1 {
    width: 598px;
    background-image: url(/images/legend-1-light.png)
  }
  .img.legend-1 div { padding-top: 48.829431% }

  :root.dark .img.legend-1 {
    background-image: url(/images/legend-1-dark.png)
  }

  .img.legend-2 {
    width: 596px;
    background-image: url(/images/legend-2-light.png)
  }
  .img.legend-2 div { padding-top: 49.328859% }

  :root.dark .img.legend-2 {
    background-image: url(/images/legend-2-dark.png)
  }

  .img.legend-3 {
    width: 597px;
    background-image: url(/images/legend-3-light.png)
  }
  .img.legend-3 div { padding-top: 48.911223% }

  :root.dark .img.legend-3 {
    background-image: url(/images/legend-3-dark.png)
  }

  .img.tooltip-1 {
    width: 596px;
    background-image: url(/images/tooltip-1-light.png)
  }
  .img.tooltip-1 div { padding-top: 48.489933% }

  :root.dark .img.tooltip-1 {
    background-image: url(/images/tooltip-1-dark.png)
  }

  .img.tooltip-2 {
    width: 960px;
    background-image: url(/images/tooltip-2-light.png)
  }
  .img.tooltip-2 div { padding-top: 47.916667% }

  :root.dark .img.tooltip-2 {
    background-image: url(/images/tooltip-2-dark.png)
  }

  .img.tooltip-treemap {
    width: 1064px;
    background-image: url(/images/tooltip-treemap-light.jpg)
  }
  .img.tooltip-treemap div { padding-top: 40.225564% }

  :root.dark .img.tooltip-treemap {
    width: 1050px;
    background-image: url(/images/tooltip-treemap-dark.jpg)
  }
  :root.dark .img.tooltip-treemap div { padding-top: 41.333333% }

  .img.tooltip-3 {
    width: 596px;
    background-image: url(/images/tooltip-3-light.png)
  }
  .img.tooltip-3 div { padding-top: 48.489933% }

  :root.dark .img.tooltip-3 {
    background-image: url(/images/tooltip-3-dark.png)
  }

  .img.tooltip-4 {
    width: 606px;
    background-image: url(/images/tooltip-4-light.png)
  }
  .img.tooltip-4 div { padding-top: 38.283828% }

  :root.dark .img.tooltip-4 {
    background-image: url(/images/tooltip-4-dark.png)
  }

  .img.tooltip-5 {
    width: 616px;
    background-image: url(/images/tooltip-5-light.png)
  }
  .img.tooltip-5 div { padding-top: 39.61039% }

  :root.dark .img.tooltip-5 {
    background-image: url(/images/tooltip-5-dark.png)
  }

  .img.tooltip-6 {
    width: 678px;
    background-image: url(/images/tooltip-6-light.png)
  }
  .img.tooltip-6 div { padding-top: 33.628319% }

  :root.dark .img.tooltip-6 {
    background-image: url(/images/tooltip-6-dark.png)
  }

  .img.tooltip-7 {
    width: 678px;
    background-image: url(/images/tooltip-7-light.png)
  }
  .img.tooltip-7 div { padding-top: 33.628319% }

  :root.dark .img.tooltip-7 {
    background-image: url(/images/tooltip-7-dark.png)
  }

  .img.tooltip-8 {
    width: 674px;
    background-image: url(/images/tooltip-8-light.png)
  }
  .img.tooltip-8 div { padding-top: 34.124629% }

  :root.dark .img.tooltip-8 {
    background-image: url(/images/tooltip-8-dark.png)
  }

  .img.tooltip-9 {
    width: 692px;
    background-image: url(/images/tooltip-9-light.png)
  }
  .img.tooltip-9 div { padding-top: 33.526012% }

  :root.dark .img.tooltip-9 {
    background-image: url(/images/tooltip-9-dark.png)
  }

  .img.tooltip-10 {
    width: 696px;
    background-image: url(/images/tooltip-10-light.png)
  }
  .img.tooltip-10 div { padding-top: 17.816092% }

  :root.dark .img.tooltip-10 {
    width: 684px;
    background-image: url(/images/tooltip-10-dark.png)
  }
  :root.dark .img.tooltip-10 div { padding-top: 15.789474% }
</style>
