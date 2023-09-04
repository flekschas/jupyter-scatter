# Selections

A primary way to interact with a scatter plot is through selections.
Jupyter Scatter makes this easy by offering selections that synchronize
between the Python and JavaScript kernel.

There are two ways to select points in a scatter plot:

1. [Programmatically using the `scatter.selection()` method](#set-programmatic)
2. [Interactively using the mouse-based lasso](#set-interactive)

Similarly, to act upon selected data points, you can

1. [Use the `scatter.selection()` method](#get-programmatic)
2. [Observe `scatter.widget.selection`](#get-interactive)

---

To demonstrate how these approaches work, we're going to use the following
DataFrame of random value:

```python
import jscatter
import numpy as np
import pandas as pd

df = pd.DataFrame({
  # Random floats
  "mass": np.random.rand(500),
  "speed": np.random.rand(500),
  "pval": np.random.rand(500),
  # Random letters A, B, C, D, E, F, G, H
  "cat": np.vectorize(lambda x: chr(65 + round(x * 8)))(np.random.rand(500)),
})

scatter = jscatter.Scatter(data=df, x="mass", y="speed", color_by="cat")
scatter.show()
```

## Programmatically Select Points {#set-programmatic}

To select a specific set of points, you can use the [`scatter.selection()`](./api#scatter.selection) method
which accepts as input a list of point indices.

```py
scatter.selection([1, 2, 3])
```

With the help of Panda's `query` method, we can easily select specific points
matching some criteria as follows:

```py
scatter.selection(df.query("cat == 'A'").index)
```

With the above call, for instance, we would select all points that belong to
category `A`.

::: info
By default, Jupyter Scatter references points by their range index. Meaning,
`scatter.selection([0, 1, 2])` will select the first, second, and third point.

Alternatively, if you're binding a DataFrame to a `Scatter` instance via
`Scatter(data=df)` and your DataFrame has a custom index, you can make the
`Scatter` instance reference data points by the DataFrame's index via
`Scatter(data=df, data_use_index=True)`.
:::

## Lasso Select Points {#set-interactive}

As you might have seen already in the [interactions guide](./interactions), we
can also select points interactively using the lasso tool.

<video autoplay loop muted playsinline width="1256" data-name="interactions-lasso">
  <source
    src="/videos/interactions-lasso-dark.mp4"
    type="video/mp4"
  />
</video>

## Get Selected Points {#get-programmatic}

Importantly, once you have selected some points, you can retrieve the
interaction using the same method [`scatter.selection()`](./api#scatter.selection)
that we used earlier. This time just don't pass any arguments to the function.

```py
scatter.selection()
# => [0, 1, 2]
```

This will return a the indices of the selected points.

If you have bound a DataFrame to the scatter instance, you can use these indices
to retrieve the original data records.

```py{2}
scatter.selection(df.query("cat == 'A'").index)
df.loc[scatter.selection()]
```

|      | x    | y    | pval | cat |
|------|------|------|------|-----|
| 0    | 0.13 | 0.27 | 0.51 | A   |
| 42   | 0.87 | 0.93 | 0.80 | A   |
| …    | …    | …    | …    | …   |
| 1337 | 0.10 | 0.25 | 0.25 | A   |

## Observe Selected Points {#get-interactive}

Real _magic_ can about to happen when you react to selections automatically. You
can do this by observing scatter widget's `selection` property:

```py{1,17-23}
import ipywidgets
import jscatter
import numpy as np
import pandas as pd

df = pd.DataFrame({
  # Random floats
  "mass": np.random.rand(500),
  "speed": np.random.rand(500),
  "pval": np.random.rand(500),
  # Random letters A, B, C, D, E, F, G, H
  "cat": np.vectorize(lambda x: chr(65 + round(x * 8)))(np.random.rand(500)),
})

scatter = jscatter.Scatter(data=df, x="mass", y="speed", color_by="cat")

output = ipywidgets.Output()

@output.capture(clear_output=True)
def selection_change_handler(change):
    display(df.loc[change.new].style.hide(axis='index'))
            
scatter.widget.observe(selection_change_handler, names=["selection"])

ipywidgets.HBox([scatter.show(), output])
````

<video autoplay loop muted playsinline width="1258" data-name="selections-observe">
  <source
    src="/videos/selections-observe-dark.mp4"
    type="video/mp4"
  />
</video>

If you want to learn how the point selections can be used to help you explore
large-scale datasets, check out our [in-depth talk+tutorial from SciPy '23](https://github.com/flekschas/jupyter-scatter-tutorial).

<script setup>
  import { videoColorModeSrcSwitcher } from './utils';
  videoColorModeSrcSwitcher();
</script>
