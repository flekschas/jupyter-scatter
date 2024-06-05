# Scales

In the following we'll go over all supported X/Y scale functions.

::: info
We'll add support for more scale functions in the future, so make sure to check
back from time to time.
:::

## Linear

By default, points are plotted linearly along the x and y coordinate.

```py
jscatter.plot(x=np.random.rand(500), y=np.random.rand(500))
```

<div class="img linear"><div /></div>

## Time

While technically identical to linear scaling, if you have temporal data, you
can render out axes with nice tick marks by setting the X or Y scale to `time`.

```py{8}
jscatter.plot(
  x=np.random.randint(
    low=1672549200000, # Jan 1, 2023 00:00:00
    high=1704085200000, # Jan 1, 2024 00:00:00
    size=500
  ),
  y=np.random.rand(500),
  x_scale='time',
)
```

<div class="img time"><div /></div>

::: warning
For the time scale to work, the data needs to be in the form of [timestamps given as the number of milliseconds since the beginning of the Unix epoch](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date#the_epoch_timestamps_and_invalid_date)!
:::

## Log

If your data is following, you can plot points 

```py{4}
jscatter.plot(
  x=np.random.rand(500),
  y=np.random.rand(500),
  x_scale='log',
)
```

<div class="img log"><div /></div>

## Power

Similarly, you can also plot points according to a power scale along the x or y
axis.

```py{4}
jscatter.plot(
  x=np.random.rand(500),
  y=np.random.rand(500),
  x_scale='pow',
)
```

<div class="img pow"><div /></div>

<style scoped>
  .img {
    max-width: 100%;
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
  }

  .img.linear {
    width: 460px;
    background-image: url(/images/scale-linear-light.png)
  }
  .img.linear div { padding-top: 56.52173913% }

  :root.dark .img.linear {
    background-image: url(/images/scale-linear-dark.png)
  }

  .img.time {
    width: 460px;
    background-image: url(/images/scale-time-light.png)
  }
  .img.time div { padding-top: 56.52173913% }

  :root.dark .img.time {
    background-image: url(/images/scale-time-dark.png)
  }

  .img.log {
    width: 460px;
    background-image: url(/images/scale-log-light.png)
  }
  .img.log div { padding-top: 56.52173913% }

  :root.dark .img.log {
    background-image: url(/images/scale-log-dark.png)
  }

  .img.pow {
    width: 460px;
    background-image: url(/images/scale-pow-light.png)
  }
  .img.pow div { padding-top: 56.52173913% }

  :root.dark .img.pow {
    background-image: url(/images/scale-pow-dark.png)
  }
</style>
