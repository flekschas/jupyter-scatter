# Tooltip

To further aid in making sense of the data points and patterns in a scatter
plot, Jupyter Scatter supports a tooltip that can show a point's properties and
related media to facilitate sense making.

```py
scatter.tooltip(True)
```

<div class="img tooltip-1"><div /></div>

Each row in the tooltip corresponds to a property. From left to right, each
property features the:

1. visual property (like `x`, `y`, `color`, `opacity`, or `size`) or data property
2. name as specified by the column name in the bound DataFrame
3. actual data value
4. histogram or treemap of the property distribution

<div class="img tooltip-2"><div /></div>

For numerical properties, the histogram is visualized as a bar chart. For
categorical properties, the histogram is visualized as a
[treemap](https://en.wikipedia.org/wiki/Treemapping) where the rectangles
represents the proportion of categories compared to the whole. Treemaps are
useful in scenarios with a lot of categories as shown below.

<div class="img tooltip-treemap"><div /></div>

In both cases, the highlighted bar / rectangle indicates how the hovered point
compares to the other points.

::: info
For demos of how to use tooltips with a variety of data, see https://github.com/flekschas/jupyter-scatter-tutorial.
:::

## Properties

By default, the tooltip shows all properties that are visually encoded but you
can limit which properties are shown:

```py
scatter.tooltip(properties=["color", "opacity"])
```

<div class="img tooltip-3"><div /></div>

Importantly, you can also show other properties in the tooltip that are not
directly visualized with the scatter plot. Other properties have to be
referenced by their respective column names.

```py{5-6}
scatter.tooltip(
  properties=[
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
The order of `properties` defines the order of the entries in the tooltip.
:::

## Customize Numerical Histograms

The histograms of numerical data properties consists of `20` bins, by default,
and is covering the entire data range, i.e., it starts at the minumum and ends
at the maximum value. You can adjust both aspects either globally for all
histograms as follows:

```py
scatter.tooltip(histograms_bins=40, histograms_ranges=(0, 1))
```

<div class="img tooltip-5"><div /></div>

To customize the number of bins and the range by property you can do:

```py
scatter.tooltip(
  histograms_bins={"color": 10, "effect_size": 30},
  histograms_ranges={"color": (0, 1), "effect_size": (0.25, 0.75)}
)
```

<div class="img tooltip-6"><div /></div>

Since an increased number of bins can make it harder to read the histogram, you
can also adjust the size as follows:

```py
scatter.tooltip(histograms_size="large")
```

<div class="img tooltip-7"><div /></div>

If you set the histogram range to be smaller than the data extent, some points
might lie outside the histogram. For instance, previously we restricted the
`effect_size` to `[0.25, 0.75]`, meaning we disregarded part of the lower and
upper end of the data.

In this case, hovering a point with an `effect_size` less than `.25` will be
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
scatter.tooltip(properties=['effect_size_winsorized'])
```

<div class="img tooltip-10"><div /></div>

## Media Previews

In cases where your data has a media representation like text, images, or audio,
you can show a preview of the media in the tooltip by referencing a column name
that holds either plain text, URLs referencing images, or URLs referencing
audio.


```py
scatter.tooltip(preview="headline")
```

<div class="img tooltip-11"><div /></div>

By default, the media type is set to `text`. If you want to show an image or
audio file as the preview, you additionally need to specify the corresponding
media type.

```py
scatter.tooltip(preview="url", preview_type="image")
```

<div class="img tooltip-12"><div /></div>

You can further customize the media preview via media type-specific arguments.
For instance in the following, we limit the audio preview to 2 seconds and
loop the audio playback.

```py
scatter.tooltip(
  preview="audio_url",
  preview_type="audio",
  preview_audio_length=2,
  preview_audio_loop=True
)
```

<div class="video">
  <video loop playsinline width="1256" data-name="tooltip-preview-audio">
    <source
      src="/videos/tooltip-preview-audio-light.mp4"
      type="video/mp4"
    />
  </video>
  <div class="overlay">Hover to play video and turn on audio</div>
</div>

For more details on how to customize the tooltip preview, see the API docs for
[`tooltip()`](/api#scatter.tooltip).

<style scoped>
  .img {
    max-width: 100%;
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
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

  .img.tooltip-11 {
    width: 814px;
    background-image: url(/images/tooltip-11-light.jpg)
  }
  .img.tooltip-11 div { padding-top: 35.87223587% }

  :root.dark .img.tooltip-11 {
    width: 764px;
    background-image: url(/images/tooltip-11-dark.jpg)
  }
  :root.dark .img.tooltip-11 div { padding-top: 37.17277487% }

  .img.tooltip-12 {
    width: 704px;
    background-image: url(/images/tooltip-12-light.png)
  }
  .img.tooltip-12 div { padding-top: 46.02272727% }

  :root.dark .img.tooltip-12 {
    width: 702px;
    background-image: url(/images/tooltip-12-dark.png)
  }
  :root.dark .img.tooltip-12 div { padding-top: 49.57264957% }

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
