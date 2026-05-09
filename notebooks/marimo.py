# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jupyter-scatter",
#     "marimo",
#     "numpy",
#     "pandas",
#     "polars",
#     "pyarrow",
#     "duckdb",
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
    import polars as pl
    import pyarrow as pa
    import duckdb

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
    return df, duckdb, jscatter, np, pa, pd, pl


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point Ordering

    Use `scatter.order()` to control the draw order of points without
    reordering the underlying data. Points drawn later appear on top.
    """)
    return


@app.cell
def _(np, pd):
    # Three overlapping gaussian blobs + a wide noise cluster with NaN category
    rng = np.random.default_rng(42)
    n_per_blob = 500

    blob_a = rng.normal(loc=[-0.5, 0.5], scale=0.4, size=(n_per_blob, 2))
    blob_b = rng.normal(loc=[0.5, 0.5], scale=0.4, size=(n_per_blob, 2))
    blob_c = rng.normal(loc=[0.0, -0.4], scale=0.4, size=(n_per_blob, 2))
    blob_noise = rng.normal(loc=[0.0, 0.2], scale=1.2, size=(n_per_blob, 2))

    coords = np.vstack([blob_a, blob_b, blob_c, blob_noise])
    categories = (
        ['A'] * n_per_blob
        + ['B'] * n_per_blob
        + ['C'] * n_per_blob
        + [None] * n_per_blob
    )

    df_blobs = pd.DataFrame(
        {
            'x': coords[:, 0],
            'y': coords[:, 1],
            'cluster': pd.Categorical(categories),
        }
    )
    # Shuffle so the default order is a mix of all clusters
    df_blobs = df_blobs.sample(frac=1, random_state=42).reset_index(drop=True)
    return (df_blobs,)


@app.cell
def _(df_blobs, jscatter):
    _kw = dict(
        data=df_blobs,
        x='x',
        y='y',
        color_by='cluster',
        size=4,
        height=280,
        legend=True,
        axes=False,
    )

    scatter_default = jscatter.Scatter(**_kw)
    scatter_c_top = jscatter.Scatter(**_kw, order_by='cluster', order_na_values='first')
    scatter_a_top = jscatter.Scatter(
        **_kw, order_by='cluster', order_direction='desc', order_na_values='first'
    )
    scatter_b_top = jscatter.Scatter(
        **_kw,
        order_by='cluster',
        order_map=['A', 'C', 'B'],
        order_na_values='first',
    )
    scatter_na_behind = jscatter.Scatter(
        **_kw, order_by='cluster', order_na_values='last'
    )

    jscatter.compose(
        [
            (scatter_default, 'Default'),
            (scatter_c_top, 'C on top'),
            (scatter_a_top, 'A on top'),
            (scatter_b_top, 'B on top'),
            (scatter_na_behind, 'NaN on top'),
        ],
        sync_selection=True,
        sync_view=True,
        rows=1,
        row_height=280,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DataFrame Input Test

    The same scatter from **Pandas**, **Polars**, **PyArrow**, and **DuckDB**.
    All four should look identical.
    """)
    return


@app.cell
def _(df, np, pa, pl):
    x = df['x'].values
    y = df['y'].values
    value = df['value'].values
    group = np.array(df['category'].values.astype(str).tolist())

    df_polars = pl.DataFrame(
        {
            'x': x,
            'y': y,
            'value': value,
            'group': group,
        }
    ).cast({'group': pl.Categorical})

    table_arrow = pa.table(
        {
            'x': x,
            'y': y,
            'value': value,
            'group': pa.DictionaryArray.from_arrays(
                pa.array(np.searchsorted(np.unique(group), group), type=pa.int32()),
                pa.array(np.unique(group)),
            ),
        }
    )
    return df_polars, table_arrow


@app.cell
def _(df, df_polars, jscatter, table_arrow):
    scatter_pd = jscatter.Scatter(
        data=df,
        x='x',
        y='y',
        color_by='category',
        size_by='value',
        legend=True,
        height=280,
    )
    scatter_pl = jscatter.Scatter(
        data=df_polars,
        x='x',
        y='y',
        color_by='group',
        size_by='value',
        legend=True,
        height=280,
    )
    scatter_pa = jscatter.Scatter(
        data=table_arrow,
        x='x',
        y='y',
        color_by='group',
        size_by='value',
        legend=True,
        height=280,
    )

    jscatter.compose(
        [
            (scatter_pd, 'Pandas'),
            (scatter_pl, 'Polars'),
            (scatter_pa, 'PyArrow'),
        ],
        sync_selection=True,
        sync_view=True,
        rows=1,
        row_height=280,
    )
    return scatter_pd, scatter_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DuckDB Query Result
    """)
    return


@app.cell
def _(duckdb, jscatter):
    conn = duckdb.connect()
    conn.execute('CREATE TABLE points AS SELECT * FROM df')
    table_duckdb = conn.sql('SELECT * FROM points').arrow()

    scatter_db = jscatter.Scatter(
        data=table_duckdb,
        x='x',
        y='y',
        color_by='category',
        size_by='value',
        legend=True,
        height=320,
    )
    scatter_db
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selection Roundtrip

    Verify that integer indices from a Polars-backed scatter match the
    Pandas-backed scatter and can index directly into the original Polars DF.
    """)
    return


@app.cell
def _(df, df_polars, mo, np, scatter_pd, scatter_pl):
    _indices = [0, 10, 50, 100]
    scatter_pd.selection(_indices)
    scatter_pl.selection(_indices)

    _sel_pd = scatter_pd.selection()
    _sel_pl = scatter_pl.selection()

    _rows_pd = df.iloc[_sel_pd]
    _rows_pl = df_polars[_sel_pl]

    mo.md(f"""
    **Pandas selection:** `{_sel_pd}`
    **Polars selection:** `{_sel_pl}`
    **Identical:** `{np.array_equal(_sel_pd, _sel_pl)}`

    **Pandas rows:**
    {mo.as_html(_rows_pd[['x', 'y', 'category']])}

    **Polars rows:**
    {mo.as_html(_rows_pl.select('x', 'y', 'group'))}
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == '__main__':
    app.run()
