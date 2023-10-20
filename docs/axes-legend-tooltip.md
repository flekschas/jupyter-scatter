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

```py{21-23}
import jscatter
import numpy as np
import pandas as pd

df = pd.DataFrame({
  # Random floats
  "mass": np.random.rand(500),
  "speed": np.random.rand(500),
  "pval": np.random.rand(500),
  "effect_size": np.random.rand(500),
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
4. histogram of the data property

<div class="img tooltip-2"><div /></div>

For numerical properties, the histogram is visualized as a bar chart. For
categorical properties, the histogram is visualized as a horizontal stacked bar.
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
    "effect_size",
    "group"
  ]
)
```

<div class="img tooltip-4"><div /></div>

Here, for instance, we're showing the point's `effect_size` and `group`
property.

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

  .img.tooltip-3 {
    width: 596px;
    background-image: url(/images/tooltip-3-light.png)
  }
  .img.tooltip-3 div { padding-top: 48.489933% }

  :root.dark .img.tooltip-3 {
    background-image: url(/images/tooltip-3-dark.png)
  }

  .img.tooltip-4 {
    width: 596px;
    background-image: url(/images/tooltip-4-light.png)
  }
  .img.tooltip-4 div { padding-top: 48.489933% }

  :root.dark .img.tooltip-4 {
    background-image: url(/images/tooltip-4-dark.png)
  }
</style>
