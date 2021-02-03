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

```python
import jscatter
import numpy as np

# Let's generate some dummy data
points = np.random.rand(500, 2)
values = np.random.rand(500) # optional
categories = (np.random.rand(500) * 10).astype(int) # optional

# Let's plot the data
scatterplot = jscatter.plot(points, categories, values)
scatterplot.show()
```

To adjust the scatter plot interactively let's pull up some options:

```python
scatterplot.options()
```

<details><summary>Click here to see options menu.</summary>
<p>

![Option UI elements](https://user-images.githubusercontent.com/932103/106693338-3f8a4400-65a4-11eb-9f4f-dd8958375709.png)

</p>
</details>

Finally, to retrieve the current selection of points (or programmatically select points) you can work with:

```python
scatterplot.selected_points
```

For a complete example, take a look at [notebooks/example.ipynb](notebooks/example.ipynb)

## API

_Coming soon!_

<details><summary>Meaningwhile type <code>jscatter.plot(</code> and hit <kbd>SHIFT</kbd>+<kbd>TAB</kbd></summary>
<p>

![Show plot options](https://user-images.githubusercontent.com/932103/106694634-f091de00-65a6-11eb-9540-928e0b6834dd.gif)


</p>
</details>

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
