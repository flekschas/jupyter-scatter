# Jupyter Test Environment

This directory contains several minimal [Conda]() environments for testing the extension in Jupyter Lab and Notebook.

- Python v3.9 with ipywidgets v7 and Jupyter Lab widgets v1
- Python v3.9 with ipywidgets v8 and Jupyter Lab widgets v3
- Python v3.10 with ipywidgets v8 and Jupyter Lab widgets v1
- Python v3.10 with ipywidgets v8 and Jupyter Lab widgets v3

1. **Install and activate the environment**

  ```bash
  conda env create -f ./environment-py39-ipyw7-jlab1.yml 
  conda activate jscatter-py39-ipyw7-jlab1
  ```

2. **Start Jupyter Lab**

  ```bash
  jupyter-lab
  ```

3. **Go to the following two pages and run the notebook**

  - [http://localhost:8888/lab/tree/test.ipynb](http://localhost:8888/lab/tree/test.ipynb)
  - [http://localhost:8888/notebooks/test.ipynb](http://localhost:8888/notebooks/test.ipynb)
