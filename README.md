<h1 align="center">
  jupyter-scatter
</h1>

<div align="center">
  
  **A scalable scatter plot extension for Jupyter Lab and Notebook**
  
</div>

<br/>

<div align="center">
  
  ![Feb-01-2021 21-31-44](https://user-images.githubusercontent.com/932103/106544399-7a717680-64d5-11eb-8d04-288b70807bc0.gif)
  
</div>

**Why?** After embedding data we want to explore the embedding space, which typically involves three things besides plotting the data as a 2D scatter. First, we want to interactively adjust the view (e.g., via panning & zooming) and the visual point encoding (e.g., the point color, opacity, or size). Second, we want to be able to select/highlight points. And third, we want to compare multiple embeddings (e.g., via animation, color, or point connections). The goal of jscatter is to support all three requirements and scale to millions of points.

**How?** Internally, jupyter-scatter is using [regl-scatterplot](https://github.com/flekschas/regl-scatterplot/) for rendering and [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) for linking the scatter plot to the iPython kernel.

## Install

```bash
pip install jupyter-scatter
```

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

Finally, to retrieve the current selection of points (or programmatically select points) you can work with:

```python
scatterplot.selected_points
```

For a complete example, take a look at [notebooks/example.ipynb](notebooks/example.ipynb)

---

## Development

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
