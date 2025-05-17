import io

import pandas as pd
import pyarrow as pa


# Code extracted from maartenbreddels ipyvolume
def df_to_arrow_ipc_buffer(df, obj=None, force_contiguous=True):
    if df is None:
        return None

    # Convert pandas DataFrame to Arrow Table
    table = pa.Table.from_pandas(df)

    # Create an in-memory buffer
    buf = io.BytesIO()

    # Write to the buffer in Arrow IPC format
    with pa.ipc.new_file(buf, table.schema) as writer:
        writer.write_table(table)

    # Get the buffer content
    buf.seek(0)

    return buf.getbuffer()


def arrow_ipc_buffer_to_df(ipc_buffer, obj=None):
    if ipc_buffer is None:
        return None

    buf_reader = pa.BufferReader(ipc_buffer)
    reader = pa.ipc.open_file(buf_reader)
    table = reader.read_all()
    return table.to_pandas()


serialization = dict(to_json=df_to_arrow_ipc_buffer, from_json=arrow_ipc_buffer_to_df)
