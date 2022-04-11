<h1 align="center">
  jupyter-scatter
</h1>

<div align="center">
  
  [![pypi version](https://img.shields.io/pypi/v/jupyter-scatter.svg?color=1a8cff&style=flat-square)](https://pypi.org/project/jupyter-scatter/)
  [![build status](https://img.shields.io/github/workflow/status/flekschas/jupyter-scatter/Build%20Python%20&%20JavaScript?color=139ce9&style=flat-square)](https://github.com/flekschas/jupyter-scatter/actions?query=workflow%3A%22Build+Python+%26+JavaScript%22)
  [![API docs](https://img.shields.io/badge/API-docs-139ce9.svg?style=flat-square)](DOCS.md)
  [![notebook examples](https://img.shields.io/badge/notebook-examples-139ce9.svg?style=flat-square)](notebooks/get-started.ipynb)
  
</div>

<div align="center">
  
  <strong>An interactive scatter plot widget for Jupyter Lab and Notebook</strong><br/>that can handle [millions of points](#visualize-millions-of-data-points) and supports [view linking](#linking-scatter-plots).
  
</div>

<br/>

<div align="center">
  
  ![Feb-01-2021 21-31-44](https://user-images.githubusercontent.com/932103/106544399-7a717680-64d5-11eb-8d04-288b70807bc0.gif)
  
</div>

**Why?** Imagine trying to explore an embedding space of millions of data points. Besides plotting the space as a 2D scatter, the exploration typically involves three things: First, we want to interactively adjust the view (e.g., via panning & zooming) and the visual point encoding (e.g., the point color, opacity, or size). Second, we want to be able to select/highlight points. And third, we want to compare multiple embeddings (e.g., via animation, color, or point connections). The goal of jupyter-scatter is to support all three requirements and scale to millions of points.

**How?** Internally, jupyter-scatter uses [regl-scatterplot](https://github.com/flekschas/regl-scatterplot/) for rendering and [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) for linking the scatter plot to the iPython kernel.

### Index

1. [Install](#install)
2. [Get Started](#get-started)
3. [API docs](API.md)
4. [Examples](notebooks/examples.ipynb)
5. [Development](#development)

## Install

```bash
pip install jupyter-scatter
```

If you are using JupyterLab <=2:

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-scatter
```

For a minimal working example, take a look at [test-environment](test-environment).

## Get Started

To play with the following examples yourself, open [notebooks/get-started.ipynb](notebooks/get-started.ipynb).

#### Simplest Example

In the simplest case, you can pass the x/y coordinates to the plot function as follows:

```python
import jscatter
import numpy as np

x = np.random.rand(500)
y = np.random.rand(500)

jscatter.plot(x, y)
```

<img width="448" alt="Simplest scatter plotexample" src="https://user-images.githubusercontent.com/932103/116143120-bc5f2280-a6a8-11eb-8614-51def74d692e.png">

#### Pandas Example

Say your data is stored in a Pandas dataframe like the following:

```python
import pandas as pd

# Just some random float and int values
data = np.random.rand(500, 4)
df = pd.DataFrame(data, columns=['mass', 'speed', 'pval', 'group'])
# We'll convert the `group` column to strings to ensure it's recognized as
# categorical data. This will come in handy in the advanced example.
df['group'] = df['group'].map(lambda c: chr(65 + round(c)), na_action=None)
```

|   | x    | y    | value | group |
|---|------|------|-------|-------|
| 0 |	0.13 | 0.27 | 0.51  | G     |
| 1 |	0.87 | 0.93 | 0.80  | B     |
| 2 |	0.10 | 0.25 | 0.25  | F     |
| 3 |	0.03 | 0.90 | 0.01  | G     |
| 4 |	0.19 | 0.78 | 0.65  | D     |

You can then visualize this data by referencing column names:

```python
jscatter.plot(data=df, x='mass', y='speed')
```

<details><summary>Show the resulting scatter plot</summary>
<img width="448" alt="Pandas scatter plot example" src="https://user-images.githubusercontent.com/932103/116143383-1364f780-a6a9-11eb-974c-4facec249974.png">
</details>

#### Advanced example

Often you want to customize the visual encoding, such as the point color, size, and opacity.

```python
jscatter.plot(
  data=df,
  x='mass',
  y='speed',
  size=8, # static encoding
  color_by='group', # data-driven encoding
  opacity_by='density', # view-driven encoding
)
```

<img width="448" alt="Advanced scatter plot example" src="https://user-images.githubusercontent.com/932103/116143470-2f689900-a6a9-11eb-861f-fcd8c563fde4.png">

In the above example, we chose a static point size of `8`. In contrast, the point color is data-driven and assigned based on the categorical `group` value. The point opacity is view-driven and defined dynamically by the number of points currently visible in the view.

Also notice how jscatter uses an appropriate color map by default based on the data type used for color encoding. In this examples, jscatter uses the color blindness safe color map from [Okabe and Ito](https://jfly.uni-koeln.de/color/#pallet) as the data type is `categorical` and the number of categories is less than `9`.

**Important:** in order for jscatter to recognize categorical data, the `dtype` of the corresponding column needs to be `category`!

You can, of course, customize the color map and many other parameters of the visual encoding as shown next.

#### Functional API Example

The [flat API](#advanced-example) can get overwhelming when you want to customize a lot of properties. Therefore, jscatter provides a functional API that groups properties by type and exposes them via meaningfully-named methods.

```python
scatter = jscatter.Scatter(data=df, x='mass', y='speed')
scatter.selection(df.query('mass < 0.5').index)
scatter.color(by='mass', map='plasma', order='reverse')
scatter.opacity(by='density')
scatter.size(by='pval', map=[2, 4, 6, 8, 10])
scatter.height(480)
scatter.background('black')
scatter.show()
```

<img width="448" alt="Functional API scatter plot example" src="https://user-images.githubusercontent.com/932103/116155554-3945c880-a6b8-11eb-9033-4d0c07f01590.png">

When you update properties dynamically, i.e., after having called `scatter.show()`, the plot will update automatically. For instance, try calling `scatter.xy('speed', 'mass')`and you will see how the points are mirrored along the diagonal.

Moreover, all arguments are optional. If you specify arguments, the methods will act as setters and change the properties. If you call a method without any arguments it will act as a getter and return the property (or properties). For example, `scatter.selection()` will return the _currently_ selected points.

Finally, the scatter plot is interactive and supports two-way communication. Hence, if you select some point with the lasso tool and then call `scatter.selection()` you will get the current selection.

#### Linking Scatter Plots

To explore multiple scatter plots and have their view, selection, and hover interactions link, use `jscatter.link()`.

```python
jscatter.link([
  jscatter.Scatter(data=embeddings, x='pcaX', y='pcaY', **config),
  jscatter.Scatter(data=embeddings, x='tsneX', y='tsneY', **config),
  jscatter.Scatter(data=embeddings, x='umapX', y='umapY', **config),
  jscatter.Scatter(data=embeddings, x='caeX', y='caeY', **config)
], rows=2)
```

https://user-images.githubusercontent.com/932103/162584133-85789d40-04f5-428d-b12c-7718f324fb39.mp4

See [notebooks/linking.ipynb](notebooks/linking.ipynb) for more details.

#### Visualize Millions of Data Points

With `jupyter-scatter` you can easily visualize and interactively explore datasets with millions of points.

In the following we're visualizing 5 million points generated with the [Rössler attractor](https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor).

```python
points = np.asarray(roesslerAttractor(5000000))
jscatter.plot(points[:,0], points[:,1], height=640)
```

https://user-images.githubusercontent.com/932103/162586987-0b5313b0-befd-4bd1-8ef5-13332d8b15d1.mp4

See [notebooks/examples.ipynb](notebooks/examples.ipynb) for more details.

---

### Development

<details><summary>Setting up a development environment</summary>
<p>

**Requirements:**

- [Conda](https://docs.conda.io/en/latest/) >= 4.8

**Installation:**

```bash
git clone https://github.com/flekschas/jupyter-scatter/ jscatter && cd jscatter
conda env create -f environment.yml && conda activate jscatter
pip install -e .
```

**Enable the Notebook Extension:**

```bash
jupyter nbextension install --py --symlink --sys-prefix jscatter
jupyter nbextension enable --py --sys-prefix jscatter
```

Note for developers: the `--symlink` argument on Linux or macOS allows one to modify
the JavaScript code in-place. This feature is not available with Windows.

**Enable the Lab Extension:**

```bash
jupyter labextension develop . --overwrite
```

**After Changing Python code:** simply restart the kernel.

**After Changing JavaScript code:** do `cd js && npm run build` and reload the browser tab.

</p>
</details>

<details><summary>Setting up a test environment</summary>
<p>

Go to [test-environment](test-environment) and follow the detailed instructions

</p>
</details>
