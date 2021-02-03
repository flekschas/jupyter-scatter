# Jupyter Test Environment

This directory contains a minimal environment to test the extension in Jupyter Lab and Notebook.

1. **Install and activate the environment**

  ```bash
  conda env create -f environment.yml
  conda activate jscatter-test
  ```

2. **Enable the extension for Jupyter Lab and Notebook**

  ```bash
  # For Jupyter Lab
  jupyter labextension install jupyter-scatter

  # For Jupyter Notebook
  jupyter nbextension install --py --sys-prefix jscatter
  jupyter nbextension enable --py --sys-prefix jscatter
  ```

3. **Start Jupyter Lab**

  ```bash
  jupyter-lab
  ```

4. **Go to the following two pages and run the notebook**

  - [http://localhost:8888/lab/tree/test.ipynb](http://localhost:8888/lab/tree/test.ipynb)
  - [http://localhost:8888/notebooks/test.ipynb](http://localhost:8888/notebooks/test.ipynb)
