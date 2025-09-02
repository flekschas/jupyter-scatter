# Interactions

Jupyter Scatter's scatter plots are interactive by default. You can pan, zoom,
hover, click, and lasso with your mouse. Moreover, Jupyter Scatter offers APIs
for filtering and zooming.

## Pan & Zoom the View

Like in Google Maps, you can pan and zoom to adjust the view and explore
different areas of the scatter plot in detail.

To pan, click and hold the primary mouse button and drag your mouse.

<div class="video">
  <video loop muted playsinline width="1256" data-name="interactions-pan">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/interactions-pan-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video</div>
</div>

To zoom, simply engage your mouse's scroll wheel.

<div class="video">
  <video loop muted playsinline width="1256" data-name="interactions-zoom">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/interactions-zoom-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video</div>
</div>

## Hover a Point

To help locate a point and where it's located, when you mouse over a point, a
reticle will appear.

<div class="video">
  <video loop muted playsinline width="1256" data-name="interactions-hover">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/interactions-hover-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video</div>
</div>

## Click a Point

In order to select a point you can click on it.

<div class="video">
  <video loop muted playsinline width="1256" data-name="interactions-click">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/interactions-click-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video</div>
</div>

## Double Click into the Void

To deselect points, simply double click into the background.

<div class="video">
  <video loop muted playsinline width="1256" data-name="interactions-double-click">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/interactions-double-click-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video</div>
</div>

## Lasso Points

To select more than a single point, you can lasso multiple points.

To activate the lasso, click and hold down your primary mouse button. An open
circle will appear and slowly close in clockwise order. Once the circle is fully
closed it'll turn blue. At this point the lasso is active.

<div class="video">
  <video loop muted playsinline width="1256" data-name="interactions-lasso">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/interactions-lasso-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video</div>
</div>

Once the lasso is active, keep holding down your primary mouse button and move
your mouse cursor around the points you want to select. Finally, release your
primary mouse key.

::: info
To permanently activate the lasso, you can click on the crosshair icon in the
top-left of the scatter plot.
:::

### Add and Remove Selected Points

To add more points to an existing selection, hold down the [_meta_ key](https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/metaKey) as you
lasso select the points you want to add. To remove points from a selection, hold
down the [_alt_ key](https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/altKey) as you lasso select the points you want to remove.

<div class="video">
  <video loop muted playsinline width="1256" data-name="interactions-lasso-add-remove">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/interactions-lasso-add-remove-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video</div>
</div>

::: info
On macOS, the meta key is `Command` or `⌘` and the alt key is `Option` or `⌥`.
:::

### Brush and Rectangle Lasso Selections

Beyond this _freeform_ lasso selection, Jupyter Scatter also offers
_rectangular_ and _brush_ style lasso selections. You can change the lasso type
either via the _lasso_ button in the top-left corner of the plot or via
`scatter.lasso(type=lasso_type)`, where `lasso_type` must be one of
`'freeform'`, `'brush'`, or `'rectangle'`.

Below is an example of the brush lasso selection.

<div class="video">
  <video loop muted playsinline width="1256" data-name="interactions-lasso-brush">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/interactions-lasso-brush-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video</div>
</div>

And finally an example of the classic rectangular lasso selection.

<div class="video">
  <video loop muted playsinline width="1256" data-name="interactions-lasso-rectangle">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/interactions-lasso-rectangle-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video</div>
</div>

## Filter Points

Sometimes it can be useful to exclusively focus on a subset of points and
temporarily hide other points. This can be achieved using [`scatter.filter()`](/api#scatter.filter):

```py
scatter.filter([0, 1, 2])
```

The filtering is blazing fast as it's only hiding non-filtered points. To unset
the filter simply call the filter function with `None`:

```py
scatter.filter(None)
```

<div class="video">
  <video loop muted playsinline width="1256" data-name="interactions-filter">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/interactions-filter-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video</div>
</div>

## Zoom to Points

When trying to focus on a subset of points (in particular point clusters), it
can help to zoom in. To zoom to a specific set of points, you can use
[`scatter.zoom()`](/api#scatter.zoom):

```py
scatter.zoom([0, 1, 2])
```

<div class="video">
  <video loop muted playsinline width="1256" data-name="interactions-zoom-to-points">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/interactions-zoom-to-points-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video</div>
</div>

You can customize how much padding you want to leave when you zoom in as follows:

```py{3}
scatter.zoom(
  to=[0, 1, 2],
  padding=1,
)
```

In this case we're instructing our scatter plot instance to have the padding be
the same size as the bounding box of the first, second, and third point we're
zooming to.

Lastly, a cool feature of Jupyter Scatter is the ability to automatically zoom
to selected or filtered points to make it as simple as possible for you to focus
on a certain set of points.

```py
scatter.zoom(on_selection=True)
```

## Overview+Detail Interface

Using everything we've learned so far and combining it with Jupyter Scatter's
ability to [synchronize/link multiple scatter plots](/link-multiple-plots)
we can easily create an overview+detail interface as follows:

```py
from jscatter import Scatter, compose
from numpy.random import rand

x, y = rand(500), rand(500)

sc_overview = Scatter(x, y)
sc_detail = Scatter(x, y, zoom_on_filter=True, zoom_padding=1)

def filter_on_select(change):
    sc_detail.filter(change['new'] if len(change['new']) > 0 else None)

sc_overview.widget.observe(filter_on_select, names='selection')

compose(
    [(sc_overview, "Overview"), (sc_detail, "Detail")],
    sync_hover=True,
)
```

<div class="video">
  <video loop muted playsinline width="826" data-name="interactions-overview-and-details">
    <source
      src="https://storage.googleapis.com/jupyter-scatter/dev/videos/interactions-overview-and-details-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video</div>
</div>

<style scoped>
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
