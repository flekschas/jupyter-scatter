# Connected Scatterplot

If your data represents series of variables, it can be useful to visualize the
evolution the variables through point connections.

::: info
Connected scatterplots are most commonly used for temporal or timeseries data.
:::

## Basics

By default, points are plotted linearly along the x and y coordinate.

```py
import datetime
import jscatter
import math
import numpy as np
import pandas as pd

def get_web_timestamp(month):
    return math.floor(
        datetime.datetime(2023, month, 1).timestamp() * 1000
    )

dates = [get_web_timestamp(month) for month in range(1, 13)]

df = DataFrame({
    'date': dates * 3,
    'value': np.concatenate(
        (
            np.random.rand(12),
            np.random.rand(12) + 0.5,
            np.random.rand(12) + 1
        ),
        axis=0
    ),
    'group': ['A'] * 12 + ['B'] * 12 + ['C'] * 12,
})

scatter = jscatter.Scatter(
    data=df,
    x='date',
    x_scale='time',
    y='value',
    connect_by='group'
)
scatter.show()
```

<div class="img basics"><div /></div>

::: tip
By default, line connections are spline interpolated curves. If you instead want
straight lines, call `scatter.options({ 'pointConnectionTolerance': 1 })`.
:::

## Styling

Similar to how you can visually style and encode the point color, opacity, and
size, you can visually style and encode the color, opacity, and size of the
point connections.

Since one might often want to use the point color as the line color, the
connection color can be set to `'inherit'` in order to inherit the color
encoding.

```py{4-6}
scatter.color(by='group')
scatter.size(10)
scatter.opacity(1)
scatter.connection_color('inherit')
scatter.connection_size(5)
scatter.connection_opacity(0.5)
```

<div class="img styled"><div /></div>

In additional, it's also possible to color the point connections by line
segments to visually distinguish the start and end of connected points.

```py
scatter.color(by='date', map='coolwarm')
scatter.connection_color(by='segment', map='coolwarm')
```

<div class="img segments"><div /></div>

<style scoped>
  .img {
    max-width: 100%;
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
  }

  .img.basics {
    width: 460px;
    background-image: url(/images/connected-scatterplot-basics-light.png)
  }
  .img.basics div { padding-top: 56.52173913% }

  :root.dark .img.basics {
    background-image: url(/images/connected-scatterplot-basics-dark.png)
  }

  .img.styled {
    width: 460px;
    background-image: url(/images/connected-scatterplot-styled-light.png)
  }
  .img.styled div { padding-top: 56.52173913% }

  :root.dark .img.styled {
    background-image: url(/images/connected-scatterplot-styled-dark.png)
  }

  .img.segments {
    width: 460px;
    background-image: url(/images/connected-scatterplot-segments-light.png)
  }
  .img.segments div { padding-top: 56.52173913% }

  :root.dark .img.segments {
    background-image: url(/images/connected-scatterplot-segments-dark.png)
  }
</style>
