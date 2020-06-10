<h1 align="center">
  jupyter-scatterplot
</h1>

<div align="center">
  
  **A lightweight but scalable scatterplot extension for Jupyter Lab and Notebook**
  
</div>

<br/>

<div align="center">
  
  [![Example](https://img.shields.io/badge/example-📖-7fd4ff.svg?style=flat-square)](https://youtu.be/FlzTdFUVE-M)
  
</div>

<div id="teaser" align="center">
  
  ![Teaser](teaser.png)
  
</div>

**Why?** After embedding data I want to explore the embedding space, which typically involves three things. First, interactive view adjustments like panning & zooming or changing the point size and opacity. Second, selecting instances. And third, coloring the embedded instances by a categorical or numerical value. The goal of jupyter-scatterplot is to easily support all three things and scale to millions of points.

**How?** Internally, jupyter-scatterplot is using [`regl-scatterplot`](https://github.com/flekschas/regl-scatterplot/)

## Development

**Requirements:**

- [Conda](https://docs.conda.io/en/latest/) >= 4.8

```bash
git clone https://github.com/flekschas/jupyter-scatterplot/ jupyter-scatterplot && cd jupyter-scatterplot
conda env create -f environment.yml && conda activate jupyter-scatterplot
make install
```
