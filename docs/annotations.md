# Annotations

To help navigating and relating points and clusters, Jupyter Scatter offers
annotations such a lines and rectangles.

::: info
Currently, Jupyter Scatter supports line-based annotations only. In the future,
we plan to add support for text annotations as well.
:::

## Line, HLine, VLine, and Rect

To draw annotations, create instances of `Line`, `HLine`, `VLine`, or `Rect`.
You can then either pass the annotations into the constructor, as shown below,
or call `scatter.annotations()`.

```py{9-17,22}
import jscatter
import numpy as np

x1, y1 = np.random.normal(-1, 0.2, 1000), np.random.normal(+1, 0.05, 1000)
x2, y2 = np.random.normal(+1, 0.2, 1000), np.random.normal(+1, 0.05, 1000)
x3, y3 = np.random.normal(+1, 0.2, 1000), np.random.normal(-1, 0.05, 1000)
x4, y4 = np.random.normal(-1, 0.2, 1000), np.random.normal(-1, 0.05, 1000)

y0 = jscatter.HLine(0)
x0 = jscatter.VLine(0)
c1 = jscatter.Rect(x_start=-1.5, x_end=-0.5, y_start=+0.75, y_end=+1.25)
c2 = jscatter.Rect(x_start=+0.5, x_end=+1.5, y_start=+0.75, y_end=+1.25)
c3 = jscatter.Rect(x_start=+0.5, x_end=+1.5, y_start=-1.25, y_end=-0.75)
c4 = jscatter.Rect(x_start=-1.5, x_end=-0.5, y_start=-1.25, y_end=-0.75)
l = jscatter.Line([
    (-2, -2), (-1.75, -1), (-1.25, -0.5), (1.25, 0.5), (1.75, 1), (2, 2)
])

scatter = jscatter.Scatter(
    x=np.concatenate((x1, x2, x3, x4)), x_scale=(-2, 2),
    y=np.concatenate((y1, y2, y3, y4)), y_scale=(-2, 2),
    annotations=[x0, y0, c1, c2, c3, c4, l],
    width=400,
    height=400,
)
scatter.show()
```

<div class="img simple"><div /></div>

## Line Color & Width

You can customize the line color and width of `Line`, `HLine`, `VLine`, or
`Rect` via the `line_color` and `line_width` attributes.

```py
y0 = jscatter.HLine(0, line_color=(0, 0, 0, 0.1))
x0 = jscatter.VLine(0, line_color=(0, 0, 0, 0.1))
c1 = jscatter.Rect(x_start=-1.5, x_end=-0.5, y_start=+0.75, y_end=+1.25, line_color="#56B4E9", line_width=2)
c2 = jscatter.Rect(x_start=+0.5, x_end=+1.5, y_start=+0.75, y_end=+1.25, line_color="#56B4E9", line_width=2)
c3 = jscatter.Rect(x_start=+0.5, x_end=+1.5, y_start=-1.25, y_end=-0.75, line_color="#56B4E9", line_width=2)
c4 = jscatter.Rect(x_start=-1.5, x_end=-0.5, y_start=-1.25, y_end=-0.75, line_color="#56B4E9", line_width=2)
l = jscatter.Line(
    [(-2, -2), (-1.75, -1), (-1.25, -0.5), (1.25, 0.5), (1.75, 1), (2, 2)],
    line_color="red",
    line_width=3
)
```

<div class="img styles"><div /></div>

<style scoped>
  .img {
    max-width: 100%;
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
  }

  .img.simple {
    width: 316px;
    background-image: url(/images/annotations-simple-light.png)
  }
  .img.simple div { padding-top: 83.2278481% }

  :root.dark .img.simple {
    background-image: url(/images/annotations-simple-dark.png)
  }

  .img.styles {
    width: 314px;
    background-image: url(/images/annotations-styles-light.png)
  }
  .img.styles div { padding-top: 83.75796178% }

  :root.dark .img.styles {
    background-image: url(/images/annotations-styles-dark.png)
  }
</style>
