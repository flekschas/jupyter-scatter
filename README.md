<h1 align="center">
  jupyter-scatter
</h1>

<div align="center">
  
  [![pypi version](https://img.shields.io/pypi/v/jupyter-scatter.svg?color=1a8cff&style=flat-square)](https://pypi.org/project/jupyter-scatter/)
  [![build status](https://img.shields.io/github/workflow/status/flekschas/jupyter-scatter/Build%20Python%20&%20JavaScript?color=139ce9&style=flat-square)](https://github.com/flekschas/jupyter-scatter/actions?query=workflow%3A%22Build+Python+%26+JavaScript%22)
  
</div>

<div align="center">
  
  **A scalable scatter plot extension for Jupyter Lab and Notebook**
  
</div>

<br/>

<div align="center">
  
  ![Feb-01-2021 21-31-44](https://user-images.githubusercontent.com/932103/106544399-7a717680-64d5-11eb-8d04-288b70807bc0.gif)
  
</div>

**IMPORTANT: THIS IS VERY EARLY WORK! THE API WILL LIKELY CHANGE.** However, you're more than welcome to give the extension and a try and let me know what you think :) All feedback is welcome!

**Why?** Imagine trying to explore an embedding space of millions of data points. Besides plotting the space as a 2D scatter, the exploration typically involves three things: First, we want to interactively adjust the view (e.g., via panning & zooming) and the visual point encoding (e.g., the point color, opacity, or size). Second, we want to be able to select/highlight points. And third, we want to compare multiple embeddings (e.g., via animation, color, or point connections). The goal of jupyter-scatter is to support all three requirements and scale to millions of points.

**How?** Internally, jupyter-scatter uses [regl-scatterplot](https://github.com/flekschas/regl-scatterplot/) for rendering and [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) for linking the scatter plot to the iPython kernel.

## Install

```bash
# Install extension
pip install jupyter-scatter

# Activate extension in Jupyter Lab
jupyter labextension install jupyter-scatter

# Activate extension in Jupyter Notebook
jupyter nbextension install --py --sys-prefix jscatter
jupyter nbextension enable --py --sys-prefix jscatter
```

For a minimal working example, take a look at [test-environment](test-environment).

## Getting Started

#### Simplest example

In the simplest case, you can pass the x/y coordinates to the plot function as follows:

```python
import jscatter
import numpy as np

x = np.random.rand(500)
y = np.random.rand(500)

jscatter.plot(x, y)
```

<details><summary>Show example</summary>
<img width="448" alt="Simplest scatter plotexample" src="https://user-images.githubusercontent.com/932103/116143120-bc5f2280-a6a8-11eb-8614-51def74d692e.png">
</details>

#### Pandas example

If your data is stored in a Pandas dataframe, you can reference columns via their name.

```python
import pandas as pd

data = np.random.rand(500, 4)
data[:,3] = np.round(data[:,3] * 7).astype(int)

df = pd.DataFrame(data, columns=['mass', 'speed', 'pval', 'group'])
df['group'] = df['group'].astype('int').astype('category').map(lambda c: chr(65 + c), na_action=None)

jscatter.plot(data=df, x='mass', y='speed')
```

<details><summary>Show example</summary>
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

<details><summary>Show example</summary>
<img width="448" alt="Advanced scatter plot example" src="https://user-images.githubusercontent.com/932103/116143470-2f689900-a6a9-11eb-861f-fcd8c563fde4.png">
</details>

In the above example, we chose a static point size of `8`. In contrast, the point color is data-driven and assigned based on the `group` value. The point opacity is view-driven and defined dynamically by the number of points currently visible in the view.

Also notice how jscatter uses an appropriate color map by default based on the data type used for color encoding. In this examples, jscatter uses the color blindness safe color map from [Okabe and Ito](https://jfly.uni-koeln.de/color/#pallet) as the number of categories is less than `9`.

You can of course customize the color map and many other parameters of the visual encoding as shown next

#### Functional API example

The [flat API](#advanced-example), can get overwhelming when you want to customize a lot of properties. Therefore, jscatter provides a functional API that groups properties by type.

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

<details><summary>Show example</summary>
<img width="448" alt="Functional API scatter plot example" src="https://user-images.githubusercontent.com/932103/116143504-398a9780-a6a9-11eb-9533-26f25a5ed788.png">
</details>

You can update properties interactively as well after having called `scatter.show()`. The plot will update automatically.

Finally, all arguments are optional. If you specify an argument, the function will act as a setter and change the property. If you call a function without any arguments it will act as a getter and return the property (or properties). For example, `scatter.selection()` will return the _currently_ selected points.

For a complete example, take a look at [notebooks/example.ipynb](notebooks/example.ipynb)

## API

### Constructor

<a name="Scatter" href="#Scatter">#</a> <b>Scatter</b>(<i>x</i>, <i>y</i>, <i>data = None</i>, <i>\*\*kwargs</i>)

**Arguments:**

- `x` is an array of quadruples defining the point data.
- `y` is an array of quadruples defining the point data.
- `data` is an array of quadruples defining the point data.
- `kwargs` is an object with the following properties:

**Returns:** a new scatter instance.

<a name="plot" href="#plot">#</a> <b>plot</b>(<i>x</i>, <i>y</i>, <i>data = None</i>, <i>\*\*kwargs</i>)

Short-hand function that creates a new scatter instance and immediately returns its widget.

**Arguments:** are the same as of [`Scatter`](#Scatter).

**Returns:** a new scatter widget.

### Methods

<a name="scatter.x" href="#scatter.x">#</a> scatter.<b>x</b>(<i>x = Undefined</i>)

Gets or sets the x coordinate.

**Arguments:**

- `x` is an array of quadruples.

**Returns:** a Promise object that resolves once the points have been drawn or transitioned.

**Examples:**

```python
scatter = Scatter('speed', 'size', data=cars)
scatter.show()
scatter.x('price') # Change x coordinates
```

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

**Enable the Lab Extension:**

```bash
jupyter labextension develop --overwrite jscatter
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
