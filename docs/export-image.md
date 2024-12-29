# Export View as Image

There are two ways to export a scatter plot as an image. You can either download
it as a PNG or save it to the widget's `view_data` property.

::: info
Image exports follow [WYSIWYG](https://en.wikipedia.org/wiki/WYSIWYG), meaning
that the exported image will have the exact same size and viewport as the
widget. Hence, if you want to export a higher resolution image you have to
increase the scatter's width and height. See [custom PNG export](#customize-png-export)
on how to easily adjust the export resolution.
:::

## Export as PNG

To download the current scatter plot view as a PNG click on the download icon.

For instance, given the following scatter.

```py
import jscatter
import numpy as np

x = np.random.normal(0, 0.1, 1000)
y = np.random.normal(0, 0.1, 1000)

scatter = jscatter.Scatter(x, y)
scatter.show()
```

<div class="img export-download"><div /></div>

The downloaded image will look as follows:

<div class="img export-download-png"><div /></div>

::: info
By default, the background color of the image is the same as
`scatter.widget.background_color`. However, you can also download the view with
a transparent background by holding down <kbd>Alt</kbd> while clicking on the
camera button.
:::

### Customize PNG Export

To better control the resolution of the exported PNG, enter the full-screen mode
(via the widget button on the left) and open up the resize panel (via the up
arrow button in the bottom-left corner). This panel allows you to easily
customize the width/height of the scatter plot and offers the ability to
up-scale the exported image.

<div class="img export-download-png-advanced"><div /></div>

::: info
Note, the exported image is subject to your [device pixel ratio](https://developer.mozilla.org/en-US/docs/Web/API/Window/devicePixelRatio), which cannot be changed. This means that if your plot is 
100 by 100 pixels in size and your display has a pixel ratio of `2`, the
exported PNG is going to be 200 by 200 pixels.
:::

## Export to `widget.view_data`

When you click on the camera icon, the current view will be exported and saved
to the widget's `view_data` property. You can use that property to print the
image or manipulate it in some way if you like.

For instance, given the following scatter.

```py
import jscatter
import numpy as np

x = np.random.rand(500)
y = np.random.rand(500)

scatter = jscatter.Scatter(x, y)
scatter.show()
```

<div class="img export-save"><div /></div>

After having clicked on the camera icon button on the left of the plot, you can
access the exported image via `scatter.widget.view_data` and, for instance, plot
it with Matplotlib as follows:

```py
from matplotlib import pyplot as plt
plt.imshow(scatter.widget.view_data, interpolation='nearest')
plt.show()
```

<div class="img export-save-matplotlib"><div /></div>

<style scoped>
  .img {
    max-width: 100%;
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
  }

  .img.export-download {
    width: 1108px;
    background-image: url(/images/export-download-light.png)
  }
  .img.export-download div { padding-top: 48.55595668% }

  :root.dark .img.export-download {
    background-image: url(/images/export-download-dark.png)
  }

  .img.export-download-png {
    width: 900px;
    background-image: url(/images/export-download-png-light.png)
  }
  .img.export-download-png div { padding-top: 53.33333333% }

  :root.dark .img.export-download-png {
    background-image: url(/images/export-download-png-dark.png)
  }

  .img.export-download-png-advanced {
    width: 3024px;
    background-image: url(/images/full-screen-light.png)
  }
  .img.export-download-png-advanced div { padding-top: 62.5% }

  :root.dark .img.export-download-png-advanced {
    background-image: url(/images/full-screen-dark.png)
  }

  .img.export-save {
    width: 1108px;
    background-image: url(/images/export-save-light.png)
  }
  .img.export-save div { padding-top: 48.55595668% }

  :root.dark .img.export-save {
    background-image: url(/images/export-save-dark.png)
  }

  .img.export-save-matplotlib {
    width: 552px;
    background-image: url(/images/export-save-matplotlib-light.png)
  }
  .img.export-save-matplotlib div { padding-top: 56.70289855% }

  :root.dark .img.export-save-matplotlib {
    background-image: url(/images/export-save-matplotlib-dark.png)
  }
</style>
