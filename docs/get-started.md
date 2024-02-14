# Get Started

## What is Jupyter Scatter?

Jupyter Scatter is a scalable, interactive, and interlinked scatter plot widget
exploring datasets with up to several million data points that runs in Jupyter
Lab/Notebook and Google Colab. It focuses on data-driven visual encodings and
offers two-way pan+zoom and lasso interactions. Beyond a single plot, Jupyter
Scatter can compose multiple scatter plots and synchronize their views and point
selections.

### Key Features

- 🖱️ **Interactive**: Pan, zoom, and select data points interactively.
- 🚀 **Scalable**: Plot up to several millions data points smoothly.
- 🔗 **Interlinked**: Synchronize the view, hover, and selection across multiple plots.
- ✨ **Effective Defaults**: Perceptually effective point colors and opacity by default.
- 📚 **Friendly API:** A readable API that integrates deeply with Pandas DataFrames.
- 🛠️ **Integratable**: Use Jupyter Scatter in your own widgets by observing its traitlets.

## Simplest Example

In the simplest case, you can pass the x/y coordinates to the plot function as follows:

```python
import jscatter
import numpy as np

x = np.random.rand(500)
y = np.random.rand(500)

jscatter.plot(x, y)
```

<div class="img get-started-simple"><div /></div>


## Bind a Pandas DataFrame

In most cases, however, it's more convenient to work with a `DataFrame` and
reference the x/y columns via their names.

```python
import jscatter
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.rand(500, 2), columns=['mass', 'speed'])

jscatter.plot(data=df, x='mass', y='speed')
```

<div class="img get-started-simple"><div /></div>

## Point Color, Opacity, and Size

Often times, we want to style the points. Jupyter Scatter allows you to do this
as follows:

```py{5-7}
jscatter.plot(
    data=df,
    x='mass',
    y='speed',
    color='red', # static visual encoding
    size=10, # static visual encoding
    opacity=0.5 # static visual encoding
)
```

<div class="img get-started-static-encoding"><div /></div>

However, more commonely, one wants to use these three point attributes (color,
opacity, and size) to visualize data properties.

For instance, in the following we're extending the data frame with a continuous
property called `pval` and a categorical property called `cat`.


```py{5,7}
df = pd.DataFrame({
  # Random floats
  "mass": np.random.rand(500),
  "speed": np.random.rand(500),
  "pval": np.random.rand(500),
  # Random letters A, B, C, D, E, F, G, H
  "cat": np.vectorize(lambda x: chr(65 + round(x * 7)))(np.random.rand(500)),
})
```

|   | x    | y    | pval | cat |
|---|------|------|------|-----|
| 0 | 0.13 | 0.27 | 0.51 | G   |
| 1 | 0.87 | 0.93 | 0.80 | B   |
| 2 | 0.10 | 0.25 | 0.25 | F   |
| 3 | 0.03 | 0.90 | 0.01 | G   |
| 4 | 0.19 | 0.78 | 0.65 | D   |

You can visualize the two properties by referencing their columns using the
`color_by`, `opacity_by`, or `size_by` arguments.

```py{5-6}
jscatter.plot(
    data=df,
    x='mass',
    y='speed',
    color_by='cat', # data-driven visual encoding
    size_by='pval', # data-driven visual encoding
)
```

<div class="img get-started-default-encoding-1"><div /></div>

Notice how `jscatter` uses a reasonable color and size map by default. Both are
based on the properties' data types. In this examples, the `jscatter` picked the
color blindness safe color map from [Okabe and Ito](https://jfly.uni-koeln.de/color/#pallet) as the number of
categories is less than 9.

When visualizing the `pval` via the color we see how the default color map
switches to Viridis given that `pval` is a continuous property.

```py{5}
jscatter.plot(
    data=df,
    x='mass',
    y='speed',
    color_by='pval', # pval is continuous data
    size_by='pval', # pval is categorical data
)
```

<div class="img get-started-default-encoding-2"><div /></div>

You can of course customize the color map and many other parameters of the visual encoding as shown next.

```py{7-19}
jscatter.plot(
    data=df,
    x='mass',
    y='speed',
    color_by='cat',
    size_by='pval',
    # Custom categorical color map 
    color_map=dict(
      A='red',    B='#00ff00', C=(0,0,1),   D='DeepSkyBlue',
      E='orange', F='#702AF7', G='#2AF7C0', H='teal'
    ),
    # Custom size map (specified as a linspace)
    size_map=(2, 20, 10),
)
```

<div class="img get-started-custom-encoding"><div /></div>

## Functional API

The flat API (that we used before) can get overwhelming when we customize a lot
of properties. Therefore, `jscatter` provides a functional API that groups
properties by type and exposes them via meaningfully-named methods that can
almost be read like a sentence.

For instance, in line two of the example below, the scatter plot colors points
by the `mass` column by mapping its values to the plasma color map in reverse
order.

```py{2}
scatter = jscatter.Scatter(data=df, x='mass', y='speed')
scatter.color(by='mass', map='plasma', order='reverse')
scatter.opacity(by='density')
scatter.size(by='pval', map=[2, 4, 6, 8, 10])
scatter.background('#1E1E20')
scatter.show()
```

<div class="img get-started-functional-api-1"><div /></div>

## Update Properties After Plotting

You don't have to specify all properties upfront. Using the functional API
you can update scatter plot instances after having plotted the scatter and the
plot will automatically re-render.

For instance, in the following we're changing the color map to `magma` in
reverse order.

```py
scatter.color(map='magma', order='reverse')
```

<div class="img get-started-functional-api-2"><div /></div>

## Chaining Method Calls

Inspired by [D3](https://d3js.org/) you can also chain methods calls as follows
to update multiple property groups at once.

```py
scatter.legend(True).axes(False)
```

<div class="img get-started-functional-api-3"><div /></div>

## Animating Point Coordinates

When you update the x/y coordinates dynamically and the number of points match,
the points will animate in a smooth transition from the previous to their new
point location.

For instance, try calling [`scatter.xy('speed', 'mass')`](./api#scatter.xy) and
you will see how the points are mirrored along the diagonal.

<video autoplay loop muted playsinline width="458" data-name="get-started-xy-animation">
  <source
    src="/videos/get-started-xy-animation-light.mp4"
    type="video/mp4"
  />
</video>

## Retrieving Properties

Lastly, all method arguments are optional. If you specify arguments, the methods
will act as setters and change the properties. However, if you call a method
without any arguments it will act as a getter and return the related properties.

For example, `scatter.color()` will return the current coloring settings.

```py
{'default': (0, 0, 0, 0.66),
 'selected': (0, 0.55, 1, 1),
 'hover': (0, 0, 0, 1),
 'by': 'mass',
 'map': [[0.001462, 0.000466, 0.013866, 1.0],
  [0.002258, 0.001295, 0.018331, 1.0],
  ...
  [0.987387, 0.984288, 0.742002, 1.0],
  [0.987053, 0.991438, 0.749504, 1.0]],
 'norm': <matplotlib.colors.Normalize at 0x15f23feb0>,
 'order': 'reverse',
 'labeling': None}
```

<style scoped>
  .img {
    max-width: 100%;
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
  }

  .img.get-started-simple {
    width: 459px;
    background-image: url(/images/get-started-simple-light.png)
  }
  .img.get-started-simple div { padding-top: 57.95207% }

  :root.dark .img.get-started-simple {
    background-image: url(/images/get-started-simple-dark.png)
  }

  .img.get-started-static-encoding {
    width: 459px;
    background-image: url(/images/get-started-static-encoding-light.png)
  }
  .img.get-started-static-encoding div { padding-top: 57.51634% }

  :root.dark .img.get-started-static-encoding {
    background-image: url(/images/get-started-static-encoding-dark.png)
  }

  .img.get-started-default-encoding-1 {
    width: 459px;
    background-image: url(/images/get-started-default-encoding-1-light.png)
  }
  .img.get-started-default-encoding-1 div { padding-top: 56.862745% }

  :root.dark .img.get-started-default-encoding-1 {
    background-image: url(/images/get-started-default-encoding-1-dark.png)
  }

  .img.get-started-default-encoding-2 {
    width: 459px;
    background-image: url(/images/get-started-default-encoding-2-light.png)
  }
  .img.get-started-default-encoding-2 div { padding-top: 56.64488% }

  :root.dark .img.get-started-default-encoding-2 {
    background-image: url(/images/get-started-default-encoding-2-dark.png)
  }

  .img.get-started-custom-encoding {
    width: 459px;
    background-image: url(/images/get-started-custom-encoding-light.png)
  }
  .img.get-started-custom-encoding div { padding-top: 57.51634% }

  :root.dark .img.get-started-custom-encoding {
    background-image: url(/images/get-started-custom-encoding-dark.png)
  }

  .img.get-started-functional-api-1 {
    width: 459px;
    background-image: url(/images/get-started-functional-api-1-light.png)
  }
  .img.get-started-functional-api-1 div { padding-top: 57.51634% }

  :root.dark .img.get-started-functional-api-1 {
    background-image: url(/images/get-started-functional-api-1-dark.png)
  }

  .img.get-started-functional-api-2 {
    width: 459px;
    background-image: url(/images/get-started-functional-api-2-light.png)
  }
  .img.get-started-functional-api-2 div { padding-top: 57.51634% }

  :root.dark .img.get-started-functional-api-2 {
    background-image: url(/images/get-started-functional-api-2-dark.png)
  }

  .img.get-started-functional-api-3 {
    width: 461px;
    background-image: url(/images/get-started-functional-api-3-light.png)
  }
  .img.get-started-functional-api-3 div { padding-top: 53.594771% }

  :root.dark .img.get-started-functional-api-3 {
    background-image: url(/images/get-started-functional-api-3-dark.png)
  }
</style>

<script setup>
  import { videoColorModeSrcSwitcher } from './utils';
  videoColorModeSrcSwitcher();
</script>
