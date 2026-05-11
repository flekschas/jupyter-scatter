import io

import pandas as pd
import pyarrow as pa


# Code extracted from maartenbreddels ipyvolume
def df_to_arrow_ipc_buffer(df, obj=None, force_contiguous=True):
    if df is None:
        return None

    # Convert to Arrow Table, supporting PyCapsule Interface (Polars, etc.)
    if isinstance(df, pd.DataFrame):
        table = pa.Table.from_pandas(df)
    else:
        table = pa.table(df)

    # Create an in-memory buffer
    buf = io.BytesIO()

    # Write to the buffer in Arrow IPC format
    with pa.ipc.new_file(buf, table.schema) as writer:
        writer.write_table(table)

    # Get the buffer content
    buf.seek(0)

    return buf.getbuffer()


def _cast_unsigned_dict_indices(table):
    """Cast unsigned dictionary indices to signed for older PyArrow compat."""
    for i, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type) and not pa.types.is_signed_integer(
            field.type.index_type
        ):
            signed_type = pa.dictionary(pa.int32(), field.type.value_type)
            table = table.set_column(i, field.name, table.column(i).cast(signed_type))
    return table


def arrow_ipc_buffer_to_df(ipc_buffer, obj=None):
    if ipc_buffer is None:
        return None

    buf_reader = pa.BufferReader(ipc_buffer)
    reader = pa.ipc.open_file(buf_reader)
    table = _cast_unsigned_dict_indices(reader.read_all())
    return table.to_pandas()


serialization = dict(to_json=df_to_arrow_ipc_buffer, from_json=arrow_ipc_buffer_to_df)
