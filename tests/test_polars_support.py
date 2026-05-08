import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from jscatter.jscatter import Scatter, check_encoding_dtype
from jscatter.dataframe_utils import ensure_pandas, _natural_sort_key
from jscatter.serializers.dataframe import (
    df_to_arrow_ipc_buffer,
    arrow_ipc_buffer_to_df,
)
from jscatter.utils import create_default_norm, to_ndc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def polars_df() -> pl.DataFrame:
    """Basic Polars DataFrame with numeric and categorical columns."""
    num_groups = 8
    np.random.seed(42)

    return pl.DataFrame(
        {
            'a': np.linspace(0, 1, 500).tolist(),
            'b': np.linspace(0, 1, 500).tolist(),
            'c': (np.random.rand(500) * 100).tolist(),
            'd': (np.random.rand(500) * 100).astype(int).tolist(),
            'group': [
                chr(65 + int(x))
                for x in np.round(np.random.rand(500) * (num_groups - 1))
            ],
            'connect': np.repeat(np.arange(100), 5).tolist(),
            'connect_order': np.resize(np.arange(5), 500).tolist(),
        }
    ).cast(
        {
            'group': pl.Categorical,
        }
    )


@pytest.fixture
def polars_df2() -> pl.DataFrame:
    """Second Polars DataFrame for update tests."""
    num_groups = 10
    np.random.seed(123)

    return pl.DataFrame(
        {
            'a': np.linspace(-2, 2, 500).tolist(),
            'b': np.linspace(-2, 2, 500).tolist(),
            'c': (np.random.rand(500) * 200).tolist(),
            'd': (np.random.rand(500) * 200).astype(int).tolist(),
            'group': [
                chr(65 + int(x))
                for x in np.round(np.random.rand(500) * (num_groups - 1))
            ],
            'connect': np.repeat(np.arange(100), 5).tolist(),
            'connect_order': np.resize(np.arange(5), 500).tolist(),
        }
    ).cast(
        {
            'group': pl.Categorical,
        }
    )


@pytest.fixture
def pandas_df(polars_df) -> pd.DataFrame:
    """Equivalent Pandas DataFrame from the same Polars data."""
    return polars_df.to_pandas()


@pytest.fixture
def arrow_table(polars_df) -> pa.Table:
    """PyArrow Table (also supports __arrow_c_stream__)."""
    return polars_df.to_arrow()


# ---------------------------------------------------------------------------
# ensure_pandas() unit tests
# ---------------------------------------------------------------------------


class TestEnsurePandas:
    def test_none_passthrough(self):
        assert ensure_pandas(None) is None

    def test_pandas_passthrough(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        result = ensure_pandas(df)
        assert result is df  # Same object, not a copy

    def test_polars_conversion(self, polars_df):
        result = ensure_pandas(polars_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 500
        assert list(result.columns) == [
            'a',
            'b',
            'c',
            'd',
            'group',
            'connect',
            'connect_order',
        ]

    def test_polars_has_range_index(self, polars_df):
        result = ensure_pandas(polars_df)
        assert isinstance(result.index, pd.RangeIndex)
        assert result.index[0] == 0
        assert result.index[-1] == 499

    def test_polars_numeric_values_preserved(self, polars_df):
        result = ensure_pandas(polars_df)
        np.testing.assert_allclose(result['a'].values, np.linspace(0, 1, 500))

    def test_polars_categorical_preserved(self, polars_df):
        result = ensure_pandas(polars_df)
        assert isinstance(result['group'].dtype, pd.CategoricalDtype)

    def test_polars_categorical_order_sorted(self, polars_df):
        """Polars orders categories by first appearance; after conversion the
        categories must be naturally sorted so that color codes are
        deterministic."""
        result = ensure_pandas(polars_df)
        cats = result['group'].cat.categories.tolist()
        assert cats == sorted(cats, key=_natural_sort_key)

    def test_polars_categorical_codes_match_native_pandas(self):
        """Category codes from Polars->ensure_pandas must match a natively
        constructed Pandas Categorical with the same values (pure alpha case
        where natural sort == lexicographic sort)."""
        values = ['C', 'A', 'B', 'A', 'C', 'B', 'D', 'D', 'A', 'B']

        # Native Pandas (sorted categories by default)
        native = pd.DataFrame({'g': pd.Categorical(values)})

        # Polars -> ensure_pandas
        converted = ensure_pandas(
            pl.DataFrame({'g': values}).cast({'g': pl.Categorical})
        )

        assert (
            native['g'].cat.categories.tolist()
            == converted['g'].cat.categories.tolist()
        )
        np.testing.assert_array_equal(
            native['g'].cat.codes.values,
            converted['g'].cat.codes.values,
        )

    def test_polars_categorical_natural_sort_order(self):
        """Categories with numbers must be sorted naturally:
        A1 < A2 < A10, not A1 < A10 < A2."""
        values = ['A10', 'A1', 'A2', 'A20', 'A3', 'A1', 'A10', 'A3']

        converted = ensure_pandas(
            pl.DataFrame({'g': values}).cast({'g': pl.Categorical})
        )

        assert converted['g'].cat.categories.tolist() == [
            'A1',
            'A2',
            'A3',
            'A10',
            'A20',
        ]

    def test_pyarrow_table_conversion(self, arrow_table):
        result = ensure_pandas(arrow_table)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 500

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match='Arrow PyCapsule Interface'):
            ensure_pandas([1, 2, 3])

    def test_unsupported_type_raises_dict(self):
        with pytest.raises(TypeError, match='Arrow PyCapsule Interface'):
            ensure_pandas({'x': [1, 2, 3]})

    def test_polars_integer_columns(self):
        df = pl.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        result = ensure_pandas(df)
        assert result['x'].tolist() == [1, 2, 3]
        assert result['y'].tolist() == [4, 5, 6]

    def test_polars_float_columns(self):
        df = pl.DataFrame({'x': [1.5, 2.5, 3.5], 'y': [4.5, 5.5, 6.5]})
        result = ensure_pandas(df)
        np.testing.assert_allclose(result['x'].values, [1.5, 2.5, 3.5])

    def test_polars_string_columns(self):
        df = pl.DataFrame({'label': ['a', 'b', 'c']})
        result = ensure_pandas(df)
        assert result['label'].tolist() == ['a', 'b', 'c']

    def test_polars_null_values(self):
        df = pl.DataFrame({'x': [1.0, None, 3.0], 'y': [None, 2.0, 3.0]})
        result = ensure_pandas(df)
        assert pd.isna(result['x'].iloc[1])
        assert pd.isna(result['y'].iloc[0])

    def test_polars_enum_dtype(self):
        """Polars Enum type should convert to Pandas Categorical."""
        df = pl.DataFrame(
            {
                'status': pl.Series(
                    ['active', 'inactive', 'active'],
                    dtype=pl.Enum(['active', 'inactive', 'pending']),
                )
            }
        )
        result = ensure_pandas(df)
        assert isinstance(result, pd.DataFrame)
        assert result['status'].tolist() == ['active', 'inactive', 'active']


# ---------------------------------------------------------------------------
# Scatter with Polars input
# ---------------------------------------------------------------------------


class TestScatterPolarsInit:
    def test_basic_scatter(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b')
        assert scatter._data is not None
        assert isinstance(scatter._data, pd.DataFrame)
        assert scatter._n == 500

    def test_widget_points_shape(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b')
        widget_data = np.asarray(scatter.widget.points)
        assert widget_data.shape == (500, 4)

    def test_widget_points_values(self, polars_df, pandas_df):
        scatter_pl = Scatter(data=polars_df, x='a', y='b')
        scatter_pd = Scatter(data=pandas_df, x='a', y='b')

        pl_points = np.asarray(scatter_pl.widget.points)
        pd_points = np.asarray(scatter_pd.widget.points)

        # Points from Polars and Pandas input should be identical
        np.testing.assert_allclose(pl_points[:, 0], pd_points[:, 0], atol=1e-6)
        np.testing.assert_allclose(pl_points[:, 1], pd_points[:, 1], atol=1e-6)

    def test_x_y_normalization(self, polars_df, pandas_df):
        scatter = Scatter(data=polars_df, x='a', y='b')
        widget_data = np.asarray(scatter.widget.points)

        expected_x = to_ndc(pandas_df['a'].values, create_default_norm())
        expected_y = to_ndc(pandas_df['b'].values, create_default_norm())

        np.testing.assert_allclose(widget_data[:, 0], expected_x)
        np.testing.assert_allclose(widget_data[:, 1], expected_y)


class TestScatterPolarsColorEncoding:
    def test_color_by_numeric(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b', color_by='c')
        widget_data = np.asarray(scatter.widget.points)
        assert np.sum(widget_data[:, 2]) > 0

    def test_color_by_categorical(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b', color_by='group')
        widget_data = np.asarray(scatter.widget.points)
        assert 'color' in scatter._encodings.visual
        assert np.sum(widget_data[:, 2]) > 0

    def test_color_categories_match_pandas(self, polars_df, pandas_df):
        scatter_pl = Scatter(data=polars_df, x='a', y='b', color_by='group')
        scatter_pd = Scatter(data=pandas_df, x='a', y='b', color_by='group')
        assert set(scatter_pl._color_categories.keys()) == set(
            scatter_pd._color_categories.keys()
        )


class TestScatterPolarsOpacitySizeEncoding:
    def test_opacity_by(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b', color_by='group')
        scatter.opacity(by='c')
        widget_data = np.asarray(scatter.widget.points)
        assert 'opacity' in scatter._encodings.visual
        assert np.sum(widget_data[:, 3]) > 0

    def test_size_by(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b', color_by='group')
        scatter.size(by='c')
        widget_data = np.asarray(scatter.widget.points)
        assert 'size' in scatter._encodings.visual
        assert np.sum(widget_data[:, 3]) > 0


class TestScatterPolarsConnectionEncoding:
    def test_connect_by(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b')
        scatter.connect(by='connect')
        widget_data = np.asarray(scatter.widget.points)
        assert widget_data.shape == (500, 5)

    def test_connect_by_with_order(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b')
        scatter.connect(by='connect', order='connect_order')
        widget_data = np.asarray(scatter.widget.points)
        assert widget_data.shape == (500, 6)


# ---------------------------------------------------------------------------
# Selection & filtering with Polars input
# ---------------------------------------------------------------------------


class TestScatterPolarsSelection:
    def test_selection_set_and_get(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b')
        _ = scatter.widget  # Instantiate widget

        scatter.selection([0, 5, 10])
        result = scatter.selection()
        np.testing.assert_array_equal(result, np.array([0, 5, 10], dtype='uint32'))

    def test_selection_clear(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b')
        _ = scatter.widget

        scatter.selection([0, 5, 10])
        scatter.selection(None)
        result = scatter.selection()
        assert len(result) == 0

    def test_filter_set_and_get(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b')
        _ = scatter.widget

        scatter.filter([1, 2, 3, 4])
        result = scatter.filter()
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4], dtype='uint32'))

    def test_filter_clear(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b')
        _ = scatter.widget

        scatter.filter([1, 2, 3])
        result = scatter.filter()
        np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype='uint32'))

        scatter.filter(None)
        # After clearing, filter returns the stored None value
        assert scatter._filtered_points_ids is None

    def test_selection_roundtrip_to_polars(self, polars_df):
        """Selected indices can be used to index back into the original Polars DF."""
        scatter = Scatter(data=polars_df, x='a', y='b')
        _ = scatter.widget

        indices = [0, 10, 50]
        scatter.selection(indices)
        selected = scatter.selection()

        # Use the integer positions to index back into the original Polars DF
        result = polars_df[selected.tolist()]
        assert len(result) == 3
        assert result['a'][0] == polars_df['a'][0]
        assert result['a'][1] == polars_df['a'][10]
        assert result['a'][2] == polars_df['a'][50]


# ---------------------------------------------------------------------------
# Data update with Polars
# ---------------------------------------------------------------------------


class TestScatterPolarsDataUpdate:
    def test_data_update_polars_to_polars(self, polars_df, polars_df2):
        scatter = Scatter(data=polars_df, x='a', y='b')
        _ = scatter.widget

        scatter.data(polars_df2)
        assert scatter._n == 500
        assert isinstance(scatter._data, pd.DataFrame)

    def test_data_update_pandas_to_polars(self, polars_df):
        pandas_df = pd.DataFrame(
            {
                'a': np.linspace(0, 1, 500),
                'b': np.linspace(0, 1, 500),
            }
        )
        scatter = Scatter(data=pandas_df, x='a', y='b')
        _ = scatter.widget

        scatter.data(polars_df)
        assert scatter._n == 500
        assert isinstance(scatter._data, pd.DataFrame)

    def test_data_update_polars_to_pandas(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b')
        _ = scatter.widget

        pandas_df = pd.DataFrame(
            {
                'a': np.linspace(-1, 1, 300),
                'b': np.linspace(-1, 1, 300),
            }
        )
        scatter.data(pandas_df)
        assert scatter._n == 300
        assert isinstance(scatter._data, pd.DataFrame)

    def test_data_update_preserves_encodings(self, polars_df, polars_df2):
        scatter = Scatter(data=polars_df, x='a', y='b', color_by='group')
        _ = scatter.widget
        assert 'color' in scatter._encodings.visual

        scatter.data(polars_df2)
        scatter.color(by='group')
        widget_data = np.asarray(scatter.widget.points)
        assert np.sum(widget_data[:, 2]) > 0


# ---------------------------------------------------------------------------
# PyArrow Table input (via PyCapsule)
# ---------------------------------------------------------------------------


class TestScatterPyArrowInput:
    def test_basic_arrow_table(self, arrow_table):
        scatter = Scatter(data=arrow_table, x='a', y='b')
        assert isinstance(scatter._data, pd.DataFrame)
        assert scatter._n == 500

    def test_arrow_points_match_polars(self, polars_df, arrow_table):
        scatter_pl = Scatter(data=polars_df, x='a', y='b')
        scatter_pa = Scatter(data=arrow_table, x='a', y='b')

        pl_points = np.asarray(scatter_pl.widget.points)
        pa_points = np.asarray(scatter_pa.widget.points)

        np.testing.assert_allclose(pl_points, pa_points, atol=1e-6)

    def test_arrow_with_color_encoding(self, arrow_table):
        scatter = Scatter(data=arrow_table, x='a', y='b', color_by='c')
        widget_data = np.asarray(scatter.widget.points)
        assert np.sum(widget_data[:, 2]) > 0


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationPolars:
    def test_polars_to_arrow_ipc(self, polars_df):
        """Polars DF serialized through ensure_pandas should round-trip via Arrow IPC."""
        pdf = ensure_pandas(polars_df)
        buf = df_to_arrow_ipc_buffer(pdf)
        assert buf is not None

        result = arrow_ipc_buffer_to_df(buf)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 500
        assert list(result.columns) == list(pdf.columns)

    def test_arrow_ipc_values_preserved(self, polars_df):
        pdf = ensure_pandas(polars_df)
        buf = df_to_arrow_ipc_buffer(pdf)
        result = arrow_ipc_buffer_to_df(buf)

        np.testing.assert_allclose(result['a'].values, pdf['a'].values)
        np.testing.assert_allclose(result['b'].values, pdf['b'].values)

    def test_serializer_pycapsule_path(self, arrow_table):
        """Arrow table can be serialized directly via the PyCapsule path."""
        buf = df_to_arrow_ipc_buffer(arrow_table)
        assert buf is not None

        result = arrow_ipc_buffer_to_df(buf)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 500


# ---------------------------------------------------------------------------
# Tooltip with Polars input
# ---------------------------------------------------------------------------


class TestScatterPolarsTooltip:
    def test_tooltip_init(self, polars_df):
        scatter = Scatter(
            data=polars_df,
            x='a',
            y='b',
            tooltip=True,
            tooltip_properties=['a', 'b', 'c', 'group'],
        )
        assert scatter.widget.tooltip_enable == True
        assert scatter.widget.tooltip_properties == ['a', 'b', 'c', 'group']

    def test_tooltip_histograms(self, polars_df, pandas_df):
        scatter = Scatter(
            data=polars_df,
            x='a',
            y='b',
            tooltip=True,
            tooltip_properties=['a', 'b'],
        )

        # X histogram should match what Pandas produces
        expected = np.histogram(pandas_df['a'].values, bins=20)[0]
        expected = expected / expected.max()
        np.testing.assert_allclose(scatter.widget.x_histogram, expected)


# ---------------------------------------------------------------------------
# Axes labels with Polars input
# ---------------------------------------------------------------------------


class TestScatterPolarsAxes:
    def test_axes_labels(self, polars_df):
        scatter = Scatter(data=polars_df, x='a', y='b')
        scatter.axes(labels=True)
        assert scatter.widget.axes_labels == ['a', 'b']


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPolarsEdgeCases:
    def test_empty_polars_df(self):
        """Empty DataFrames trigger a numpy min/max error (pre-existing issue)."""
        df = pl.DataFrame({'x': [], 'y': []}).cast({'x': pl.Float64, 'y': pl.Float64})
        result = ensure_pandas(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_row_polars_df(self):
        df = pl.DataFrame({'x': [1.0], 'y': [2.0]})
        scatter = Scatter(data=df, x='x', y='y')
        assert scatter._n == 1

    def test_large_polars_df(self):
        n = 100_000
        df = pl.DataFrame(
            {
                'x': np.random.rand(n).tolist(),
                'y': np.random.rand(n).tolist(),
            }
        )
        scatter = Scatter(data=df, x='x', y='y')
        assert scatter._n == n

    def test_polars_with_many_columns(self):
        data = {f'col_{i}': np.random.rand(100).tolist() for i in range(50)}
        data['x'] = np.random.rand(100).tolist()
        data['y'] = np.random.rand(100).tolist()
        df = pl.DataFrame(data)

        scatter = Scatter(data=df, x='x', y='y')
        assert scatter._n == 100

    def test_polars_boolean_column(self):
        df = pl.DataFrame(
            {
                'x': [1.0, 2.0, 3.0],
                'y': [4.0, 5.0, 6.0],
                'flag': [True, False, True],
            }
        )
        scatter = Scatter(data=df, x='x', y='y')
        assert scatter._n == 3

    def test_polars_multiple_categorical_columns(self):
        df = pl.DataFrame(
            {
                'x': [1.0, 2.0, 3.0, 4.0],
                'y': [4.0, 5.0, 6.0, 7.0],
                'cat_a': ['a', 'b', 'a', 'b'],
                'cat_b': ['x', 'y', 'x', 'y'],
            }
        ).cast(
            {
                'cat_a': pl.Categorical,
                'cat_b': pl.Categorical,
            }
        )
        scatter = Scatter(data=df, x='x', y='y', color_by='cat_a')
        widget_data = np.asarray(scatter.widget.points)
        assert np.sum(widget_data[:, 2]) > 0

    def test_polars_with_nan_in_encoding(self):
        no_nan = [0.0, 0.25, 0.5, 0.75, 1.0]
        with_nan = [0.0, 0.25, 0.5, None, 1.0]

        df = pl.DataFrame({'x': no_nan, 'y': no_nan, 'z': with_nan})

        with pytest.warns(UserWarning, match='missing values'):
            scatter = Scatter(data=df, x='x', y='y', color_by='z')
            assert np.isfinite(scatter.widget.points).all()

    def test_polars_datetime_column_ignored(self):
        """Datetime columns shouldn't break anything if not used for encoding."""
        from datetime import date

        df = pl.DataFrame(
            {
                'x': [1.0, 2.0, 3.0],
                'y': [4.0, 5.0, 6.0],
                'date': [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            }
        )
        scatter = Scatter(data=df, x='x', y='y')
        assert scatter._n == 3


# ---------------------------------------------------------------------------
# Custom PyCapsule-compatible object
# ---------------------------------------------------------------------------


class TestCustomPyCapsuleObject:
    def test_custom_object_with_arrow_c_stream(self):
        """Any object with __arrow_c_stream__ should work."""
        # Create a PyArrow table and use it as a PyCapsule source
        table = pa.table({'x': [1.0, 2.0, 3.0], 'y': [4.0, 5.0, 6.0]})

        # Simulate a custom object that wraps Arrow data
        class CustomArrowWrapper:
            def __init__(self, table):
                self._table = table

            def __len__(self):
                return len(self._table)

            def __arrow_c_stream__(self, requested_schema=None):
                return self._table.__arrow_c_stream__(requested_schema)

        wrapper = CustomArrowWrapper(table)
        result = ensure_pandas(wrapper)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert result['x'].tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# DuckDB integration (if available)
# ---------------------------------------------------------------------------


class TestDuckDBIntegration:
    @pytest.fixture(autouse=True)
    def _check_duckdb(self):
        pytest.importorskip('duckdb')

    def test_duckdb_result_to_scatter(self):
        import duckdb

        conn = duckdb.connect()
        result = conn.sql(
            'SELECT x::DOUBLE as x, y::DOUBLE as y FROM '
            '(VALUES (1, 4), (2, 5), (3, 6)) AS t(x, y)'
        ).arrow()

        scatter = Scatter(data=result, x='x', y='y')
        assert scatter._n == 3
        assert isinstance(scatter._data, pd.DataFrame)
