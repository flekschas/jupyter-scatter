import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..font import Font, font_name_map
from .constants import DEFAULT_ZOOM_RANGE
from .label_placement import LabelPlacement
from .types import PersistenceFormat


def _get_path_prefix(path: str) -> str:
    """
    Determine if `path` references a directory or a prefix

    Parameters
    ----------
        path : str
        Path to save data. Can be either:
        - A directory path where files 'labels.parquet' and 'tiles.parquet' will be stored
        - A path prefix where files '{prefix}-labels.parquet' and '{prefix}-tiles.parquet' will be stored

    Returns
    -------
    str
        A valid path prefix str
    """
    path_obj = Path(path)

    if path_obj.exists() and path_obj.is_dir():
        # Path exists and is a directory, use directory mode
        return f'{path}/label'

    # Check if parent directory exists
    parent_dir = path_obj.parent
    if parent_dir.exists() and parent_dir.is_dir():
        # Parent directory exists, use prefix mode
        return f'{path}-label'

    raise ValueError(
        f'Invalid path. It must either point to a directory or a file prefix in a directory: {path}'
    )


def _serialize_font(font: Font) -> str:
    """Convert a Font object to a string representation for serialization."""
    for key, value in font_name_map.items():
        if value == font:
            return key

    import warnings

    warnings.warn('Custom fonts cannot be serialized. Will use "arial" instead.')

    return 'arial'


def _deserialize_font(font: str) -> Font:
    """Convert a font str back to a Font object."""
    return font_name_map.get(font, font_name_map['arial'])


def to_parquet(
    label_placement: LabelPlacement,
    path: str,
    format: PersistenceFormat = 'parquet',
):
    """
    Export label placement data to persistence storage.

    Parameters
    ----------
    label_placement : LabelPlacement
        Label placement object to export
    path : str
        Path to save data. Can be either:
        - A directory path where files 'labels.parquet' and 'tiles.parquet' will be stored
        - A path prefix where files '{prefix}-labels.parquet' and '{prefix}-tiles.parquet' will be stored
    format : PersistenceFormat, default=PersistenceFormat.PARQUET
        Format to use for persistence
    use_prefix : bool, optional
        If True, treat path as a prefix for output files.
        If False, treat path as a directory.
        If None (default), automatically detect based on whether path exists as a directory.
    """

    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq

    if format != 'parquet' and format != 'arrow_ipc':
        raise ValueError(
            'Invalid format. Only "parquet" and "arrow_ipc" are supported.'
        )

    path_prefix = _get_path_prefix(path)

    ext = 'parquet' if format == 'parquet' else 'arrow'

    labels_path = f'{path_prefix}-data.{ext}'
    tiles_path = f'{path_prefix}-tiles.{ext}'

    if label_placement.labels is None or label_placement.tiles is None:
        raise ValueError(
            'Label placement does not have labels or tiles. You need to compute labels first.'
        )

    # Prepare metadata
    metadata = {
        'tile_size': int(label_placement.tile_size),
        '_x_extent': float(label_placement._x_extent),
        '_y_extent': float(label_placement._y_extent),
        '_x_min': float(label_placement._x_min),
        '_y_min': float(label_placement._y_min),
        '_x_max': float(label_placement._x_max),
        '_y_max': float(label_placement._y_max),
        'by': label_placement.by,
        'x': label_placement.x,
        'y': label_placement.y,
        'hierarchical': label_placement.hierarchical,
        'bbox_percentile_range': list(label_placement.bbox_percentile_range),
        'max_labels_per_tile': label_placement.max_labels_per_tile,
        'scale_function': label_placement.scale_function,
        'positioning': label_placement.positioning,
        'importance': label_placement.importance,
        'importance_aggregation': label_placement.importance_aggregation
        if isinstance(label_placement.importance_aggregation, str)
        else 'mean',
        'target_aspect_ratio': label_placement.target_aspect_ratio,
        'max_lines': label_placement.max_lines,
        'background': label_placement.background,
        'size': label_placement.size,
        'color': label_placement.color,
        'zoom_range': {
            key: list(map(float, zoom_range))
            for key, zoom_range in label_placement.zoom_range.items()
        },
        'exclude': label_placement.exclude,
        'font': {
            key: _serialize_font(font) for key, font in label_placement.font.items()
        },
    }

    if not isinstance(label_placement.importance_aggregation, str):
        import warnings

        warnings.warn(
            'Custom `importance_aggregation` function cannot be serialized. Default will be used when loaded.'
        )

    # Convert metadata to JSON for storage
    json_metadata = {b'label_placement_metadata': json.dumps(metadata).encode()}

    # Convert label data and tiles to PyArrow tables
    label_table = pa.Table.from_pandas(label_placement.labels)
    label_table = label_table.replace_schema_metadata(
        {**label_table.schema.metadata, **json_metadata}
    )

    tile_data = _tiles_to_dataframe(label_placement.tiles)
    tile_table = pa.Table.from_pandas(tile_data)

    # Handle the different formats
    if format == 'parquet':
        pq.write_table(label_table, labels_path)
        pq.write_table(tile_table, tiles_path)
    elif format == 'arrow_ipc':
        with pa.OSFile(labels_path, 'wb') as sink:
            writer = ipc.new_file(sink, label_table.schema)
            writer.write(label_table)
            writer.close()

        with pa.OSFile(tiles_path, 'wb') as sink:
            writer = ipc.new_file(sink, tile_table.schema)
            writer.write(tile_table)
            writer.close()


def from_parquet(
    path: str,
    format: Optional[PersistenceFormat] = None,
) -> LabelPlacement:
    """
    Load label placement data from persistent storage.

    Parameters
    ----------
    path : str
        Path to load data from. Can be either:
        - A directory path containing 'label-data.parquet' and 'label-tiles.parquet'
        - A path prefix pointing to '{prefix}-label-data.parquet' and '{prefix}-label-tiles.parquet'
    format : PersistenceFormat, optional
        Format to use for loading. If None, will be determined from the path.

    Returns
    -------
    LabelPlacement
        Loaded label placement object
    """

    import pandas as pd
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq

    path_prefix = _get_path_prefix(path)

    # If format is not specified, try to determine it from file extensions
    if format is None:
        for ext in ['parquet', 'arrow']:
            if Path(f'{path_prefix}-data.{ext}').exists():
                format = 'parquet' if ext == 'parquet' else 'arrow_ipc'
                break
        if format is None:
            raise FileNotFoundError(f'No files found with prefix: {path}')

    if format != 'parquet' and format != 'arrow_ipc':
        raise ValueError(
            'Invalid format. Only "parquet" and "arrow_ipc" are supported.'
        )

    ext = 'parquet' if format == 'parquet' else 'arrow'

    # Define expected file paths with prefix
    labels_path = f'{path_prefix}-data.{ext}'
    tiles_path = f'{path_prefix}-tiles.{ext}'

    # Check if files exist
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f'File not found: {labels_path}')
    if not os.path.exists(tiles_path):
        raise FileNotFoundError(f'File not found: {tiles_path}')

    # Extract data based on format
    labels = None
    tiles_df = None
    metadata = None
    if format == 'parquet':
        # Load label and tile data and extract metadata
        table = pq.read_table(labels_path)

        # Extract metadata from schema
        schema_metadata = table.schema.metadata
        if b'label_placement_metadata' in schema_metadata:
            metadata_json = schema_metadata[b'label_placement_metadata'].decode('utf-8')
            metadata = json.loads(metadata_json)
        else:
            raise ValueError('No metadata found in the label data parquet file')

        labels = table.to_pandas()

        tiles_df = pq.read_table(tiles_path).to_pandas()

    elif format == 'arrow_ipc':
        with pa.memory_map(labels_path, 'rb') as source:
            reader = ipc.open_file(source)
            table = reader.read_all()

            # Extract metadata from schema
            schema_metadata = table.schema.metadata
            if b'label_placement_metadata' in schema_metadata:
                metadata_json = schema_metadata[b'label_placement_metadata'].decode(
                    'utf-8'
                )
                metadata = json.loads(metadata_json)

            labels = table.to_pandas()

        with pa.memory_map(tiles_path, 'rb') as source:
            reader = ipc.open_file(source)
            table = reader.read_all()

            # If we haven't extracted metadata yet, try from tile data
            if (
                metadata is None
                and b'label_placement_metadata' in reader.schema.metadata
            ):
                metadata_json = reader.schema.metadata[
                    b'label_placement_metadata'
                ].decode('utf-8')
                metadata = json.loads(metadata_json)

            tiles_df = table.to_pandas()

    # Create the object with the loaded parameters
    if metadata is None:
        raise ValueError(f'Could not find metadata in the provided {format} files')

    # Create a new LabelPlacement instance with appropriate columns
    empty_data = {
        **{metadata['x']: [], metadata['y']: []},
        **{by: [] for by in metadata['by']},
    }

    # Add importance column if specified in metadata
    if metadata.get('importance'):
        empty_data[metadata['importance']] = []

    # Create a new LabelPlacement instance
    label_placement = LabelPlacement(
        # Empty DataFrame
        data=pd.DataFrame(empty_data),
        by=metadata['by'],
        x=metadata['x'],
        y=metadata['y'],
        tile_size=metadata['tile_size'],
        hierarchical=metadata['hierarchical'],
        bbox_percentile_range=tuple(metadata['bbox_percentile_range']),
        max_labels_per_tile=metadata['max_labels_per_tile'],
        scale_function=metadata['scale_function'],
        positioning=metadata['positioning'],
        importance=metadata.get('importance'),
        importance_aggregation=metadata.get('importance_aggregation', 'mean'),
        target_aspect_ratio=metadata.get('target_aspect_ratio'),
        max_lines=metadata.get('max_lines'),
        background=metadata.get('background'),
        size=metadata.get('size', 'auto'),
        color=metadata.get('color', 'auto'),
        zoom_range={
            key: tuple(zoom_range)
            for key, zoom_range in metadata.get('zoom_range', {}).items()
        },
        exclude=metadata.get('exclude', []),
        font={
            key: _deserialize_font(font)
            for key, font in metadata.get('font', {}).items()
        },
    )

    # After loading, check if any complex properties couldn't be fully restored
    if (
        metadata.get('importance_aggregation') is None
        and label_placement.importance is not None
    ):
        import warnings

        warnings.warn(
            "Custom `importance_aggregation` function couldn't be restored from serialized data."
        )

    # Set properties directly
    label_placement._x_min = metadata['_x_min']
    label_placement._x_max = metadata['_x_max']
    label_placement._y_min = metadata['_y_min']
    label_placement._y_max = metadata['_y_max']
    label_placement._x_extent = metadata['_x_extent']
    label_placement._y_extent = metadata['_y_extent']

    # Convert categorical columns to the correct type
    if labels is not None:
        # Ensure categorical columns have proper types
        for col in labels.columns:
            if col in [
                'label_type',
                'font_face',
                'font_style',
                'font_weight',
                'font_color',
                'font_size',
            ]:
                if not pd.CategoricalDtype.is_dtype(labels[col]):
                    labels[col] = pd.Categorical(labels[col])

        # Set label data
        label_placement._labels = labels

    # Import tile data
    if tiles_df is not None:
        label_placement._tiles = _tiles_from_dataframe(tiles_df)

    label_placement._loaded_from_persistence = True

    return label_placement


def _tiles_to_dataframe(tiles: Dict[str, List[int]]) -> pd.DataFrame:
    """
    Export the tile dictionary to a DataFrame for storage.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns x, y, z, and label
    """
    tile_data = []

    for tile_id, label_indices in tiles.items():
        x, y, z = map(int, tile_id.split(','))
        for label_index in label_indices:
            tile_data.append({'x': x, 'y': y, 'z': z, 'label': label_index})

    return pd.DataFrame(tile_data)


def _tiles_from_dataframe(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Import tile data from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns x, y, z, and label
    """
    tiles = {}

    for _, row in df.iterrows():
        tile_id = f'{row["x"]},{row["y"]},{row["z"]}'
        if tile_id not in tiles:
            tiles[tile_id] = []
        tiles[tile_id].append(row['label'])

    return tiles
