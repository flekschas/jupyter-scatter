<h1 align="center">
  jscatter
</h1>

<div align="center">
  
  **A lightweight but scalable scatter plot extension for Jupyter Lab and Notebook**
  
</div>

<br/>

<div align="center">
  
  [![Example](https://img.shields.io/badge/example-📖-7fd4ff.svg?style=flat-square)](https://github.com/flekschas/jscatter/blob/master/notebooks/example.ipynb)
  
</div>

<div id="teaser" align="center">
  
  ![Teaser](teaser.png)
  
</div>

**Why?** After embedding data we want to explore the embedding space, which typically involves three things besides plotting the data as a 2D scatter. First, we want to interactively adjust the view (e.g., via panning & zooming) and the visual point encoding (e.g., the point color, opacity, or size). Second, we want to be able to select/highlight points. And third, we want to compare multiple embeddings (e.g., via animation, color, or point connections). The goal of jscatter is to support all three requirements and scale to millions of points.

**How?** Internally, jscatter is using [`regl-scatterplot`](https://github.com/flekschas/regl-scatterplot/)

## Development

**Requirements:**

- [Conda](https://docs.conda.io/en/latest/) >= 4.8

```bash
git clone https://github.com/flekschas/jscatter/ jscatter && cd jscatter
conda env create -f environment.yml && conda activate jscatter
make install
```
