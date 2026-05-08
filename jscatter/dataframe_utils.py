from __future__ import annotations

import re

import pandas as pd
import pyarrow as pa

_NATURAL_SORT_RE = re.compile(r'(\d+)')


def _natural_sort_key(value):
    """Sort key that orders strings naturally: A1 < A2 < A10."""
    parts = _NATURAL_SORT_RE.split(str(value))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def ensure_pandas(data):
    """Convert a DataFrame-like object to a Pandas DataFrame.

    Accepts:
    - Pandas DataFrames (returned as-is)
    - Any object implementing the Arrow PyCapsule Interface
      (``__arrow_c_stream__``), e.g. Polars, DuckDB, cuDF
    - ``None`` (returned as-is)

    Raises
    ------
    TypeError
        If the input is not a supported DataFrame type.
    """
    if data is None:
        return data

    if isinstance(data, pd.DataFrame):
        return data

    if hasattr(data, '__arrow_c_stream__'):
        table = pa.RecordBatchReader.from_stream(data).read_all()

        # Polars exports categoricals as Arrow dictionary arrays with uint32
        # indices. Older PyArrow versions (<14) can't convert unsigned
        # dictionary indices to Pandas, so we cast them to signed int32.
        for i, field in enumerate(table.schema):
            if pa.types.is_dictionary(field.type) and not pa.types.is_signed_integer(
                field.type.index_type
            ):
                signed_type = pa.dictionary(pa.int32(), field.type.value_type)
                table = table.set_column(
                    i, field.name, table.column(i).cast(signed_type)
                )

        df = table.to_pandas()

        # Arrow dictionary columns (from Polars Categorical, etc.) become
        # Pandas Categoricals whose category order matches the source library.
        # Polars orders by first appearance; Pandas conventions expect sorted
        # categories.  Re-sorting ensures consistent color/code assignments
        # regardless of the input library.
        for col in df.columns:
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].cat.reorder_categories(
                    sorted(df[col].cat.categories, key=_natural_sort_key)
                )

        return df

    raise TypeError(
        f'Expected a Pandas or Polars DataFrame (or any object implementing '
        f'the Arrow PyCapsule Interface), got {type(data).__name__}'
    )


def is_pandas(data) -> bool:
    """Check if data is a Pandas DataFrame."""
    return isinstance(data, pd.DataFrame)
