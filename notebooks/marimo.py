# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jupyter-scatter",
#     "marimo",
#     "numpy",
#     "pandas",
# ]
#
# [tool.uv.sources]
# jupyter-scatter = { path = "..", editable = true }
# ///

import marimo

__generated_with = '0.23.5'
app = marimo.App(width='medium')


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 👋 Jupyter Scatter in Marimo
    """)
    return


@app.cell
def _():
    import jscatter
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n = 500
    df = pd.DataFrame(
        {
            'x': np.random.normal(0, 1, n),
            'y': np.random.normal(0, 1, n),
            'category': pd.Categorical(np.random.choice(['A', 'B', 'C'], n)),
            'value': np.random.uniform(0, 100, n),
        }
    )
    return df, jscatter


@app.cell
def _(df, jscatter):
    scatter = jscatter.Scatter(
        data=df, x='x', y='y', color_by='category', size_by='value', height=320
    )
    scatter
    return


@app.cell
def _(df, jscatter):
    scatter_a = jscatter.Scatter(data=df, x='x', y='y', color_by='category')
    scatter_b = jscatter.Scatter(data=df, x='x', y='value', color_by='category')
    jscatter.compose(
        [
            (scatter_a, 'X vs Y'),
            (scatter_b, 'X vs Value'),
        ],
        sync_selection=True,
        sync_view=False,
        rows=1,
        row_height=320,
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == '__main__':
    app.run()
