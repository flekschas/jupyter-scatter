# Link Multiple Scatter Plots

There are many use cases where one wants to create two or more scatter plot and
potentially link the view, selection, and hover state.

For instance, we might want to
- visualize different properties of a dataset
- compare different facets of a dataset
- compare different embedding methods of a dataset
- compare multiple datasets with shared labels

Jupyter Scatter can help with this by composing and linking multiple scatter
instances.

## Simplest Example

We'll start out with a very simple example to get familiar with the API.

In the following, we'll compose two scatter plots next to each other using
[`compose()`](./api#compose).

```py{7}
import jscatter
import numpy as np

x, y = np.random.rand(500), np.random.rand(500)
a, b = jscatter.Scatter(x=x, y=y), Scatter(x=x, y=y)

jscatter.compose([a, b])
```

<div class="img link-multiple-plots-simple-1"><div /></div>

By default, jscatter arranges scatter plots into a single row but we can
customize this of course.

```py
jscatter.compose([a, b], rows=2)
```

<div class="img link-multiple-plots-simple-2"><div /></div>

We can also change the row height as follows to shrink or grow the plots.

```py
jscatter.compose([a, b], rows=2, row_height=240)
```

<div class="img link-multiple-plots-simple-3"><div /></div>

## Synchronize View, Selection, & Hover

So good so far but the fun part starts when we link/synchronize the scatter
plots' views and selections.

```py{3-5}
jscatter.compose(
    [a, b],
    sync_view=True,
    sync_selection=True,
    sync_hover=True,
)
```

<video autoplay loop muted playsinline data-name="link-multiple-plots-synced" width="1000">
  <source
    src="/videos/link-multiple-plots-synced-light.mp4"
    type="video/mp4"
  />
</video>

Since a common use case is to synchronize everything, `jscatter` offers the
shorthand method [`link()`](./api#link):

```py
jscatter.link([a, b])
```

The result is the same as the previous [`compose()`](./api#compose) call.

<style scoped>
  .img {
    max-width: 100%;
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
  }

  .img.link-multiple-plots-simple-1 {
    width: 812px;
    background-image: url(/images/link-multiple-plots-simple-1-light.png);
  }

  .img.link-multiple-plots-simple-1 div { padding-top: 40.147783% }

  :root.dark .img.link-multiple-plots-simple-1 {
    background-image: url(/images/link-multiple-plots-simple-1-dark.png);
  }

  .img.link-multiple-plots-simple-2 {
    width: 812px;
    background-image: url(/images/link-multiple-plots-simple-2-light.png);
  }

  .img.link-multiple-plots-simple-2 div { padding-top: 79.310345% }

  :root.dark .img.link-multiple-plots-simple-2 {
    background-image: url(/images/link-multiple-plots-simple-2-dark.png);
  }

  .img.link-multiple-plots-simple-3 {
    width: 812px;
    background-image: url(/images/link-multiple-plots-simple-3-light.png);
  }

  .img.link-multiple-plots-simple-3 div { padding-top: 59.852217% }

  :root.dark .img.link-multiple-plots-simple-3 {
    background-image: url(/images/link-multiple-plots-simple-3-dark.png);
  }
</style>

<script setup>
  import { videoColorModeSrcSwitcher } from './utils';
  videoColorModeSrcSwitcher();
</script>
