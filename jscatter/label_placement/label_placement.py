import hashlib
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from geoindex_rs import kdtree, rtree
from matplotlib.colors import to_hex
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError

from ..dependencies import check_label_extras_dependencies, MissingCallable
from ..font import Font
from ..types import (
    AggregationMethod,
    Auto,
    Color,
    LabelPositioning,
    LabelScaleFunction,
    LogLevel,
)
from ..utils import calculate_luminance
from .aggregate import aggregate
from .center_of_mass import compute_center_of_mass
from .constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FONT_FACE,
    DEFAULT_FONT_SIZE,
    DEFAULT_MAX_LABELS_PER_TILE,
    DEFAULT_TILE_SIZE,
    DEFAULT_ZOOM_RANGE,
    MAX_ZOOM_LEVEL,
    ZOOM_IN_PX,
)
from .highest_density_point import compute_highest_density_point
from .largest_cluster import compute_largest_cluster
from .optimize_line_breaks import optimize_line_breaks
from .resolve_zoom_in_collisions import (
    resolve_asinh_zoom_in_collisions,
    resolve_static_zoom_in_collisions,
)
from .text import Text
from .tile import get_tile_id, zoom_level_to_zoom_scale, zoom_scale_to_zoom_level
from .types import PersistenceFormat
from .utils import (
    configure_logging,
    create_linear_scale,
    deduplicate,
    flatten,
    get_unique_labels,
    is_categorical_data,
    map_binary_list_property,
    map_property,
    noop,
    remove_line_breaks,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = MissingCallable.class_("tqdm", "tqdm.auto", "label-extras")


def is_default_zoom_range(zoom_range):
    return math.isinf(zoom_range[0]) and zoom_range[0] < 0 and math.isinf(zoom_range[1])


def is_default_zoom_ranges(zoom_ranges):
    return {
        key: is_default_zoom_range(zoom_range)
        for key, zoom_range in zoom_ranges.items()
    }


logger = logging.getLogger(__name__)


class LabelPlacement:
    """
    A class for static label placement with collision resolution.

    This class handles the positioning of labels for data points while managing
    collisions and optimizing label visibility across different zoom levels.

    Attributes
    ----------
    computed : bool
        Whether labels have been computed.
    loaded_from_persistence : bool
        Whether this instance was restored from saved files.
    background : Color
        The background color.
    color : Dict[str, str]
        A dictionary mapping each label type and specific label to its color.
    font : Dict[str, Font]
        A dictionary mapping each label type and specific label to its font face.
    size : Dict[str, int]
        A dictionary mapping each label type and specific label to its size.
    zoom_range : Dict[str, Tuple[np.float64, np.float64]]
        A dictionary mapping each label type and specific label to its zoom range.
    exclude : List[str]
        A list of label types and specific labels to be excluded.
    data : pd.DataFrame
        The input data.
    x : str
        The name of the x-coordinate column.
    y : str
        The name of the y-coordinate column.
    by : List[str]
        The column name(s) defining the label hierarchy.
    hierarchical : bool
        Whether the labels are hierarchical.
    importance : Optional[str]
        The name of the importance column.
    importance_aggregation : AggregationMethod
        The importance aggregation method.
    bbox_percentile_range : Tuple[float, float]
        The percentile range for bounding box calculation.
    tile_size : int
        The tile size.
    max_labels_per_tile : int
        The maximum number of labels per tile.
    scale_function : LabelScaleFunction
        The label zoom scale function.
    positioning : LabelPositioning
        The label positioning method.
    target_aspect_ratio : Optional[float]
        The target aspect ratio for line break optimization.
    max_lines : Optional[int]
        The maximum number of lines for line break optimization.
    verbosity : LogLevel
        The current log level.
    labels : Optional[pd.DataFrame]
        The computed labels, if available.
    tiles : Optional[Dict[str, List[int]]]
        The tile mapping, if available.
    """

    _color: Dict[str, str]
    _exclude: Set[str]
    _font: Dict[str, Font]
    _importance_aggregation: AggregationMethod
    _labels: Optional[pd.DataFrame] = None
    _max_lines: Optional[int] = None
    _point_label_columns: List[str]
    _positioning: LabelPositioning
    _scale_function: LabelScaleFunction
    _size: Dict[str, int]
    _target_aspect_ratio: Optional[float] = None
    _tiles: Optional[Dict[str, List[int]]] = None
    _verbosity: LogLevel
    _zoom_range: Dict[str, Tuple[np.float64, np.float64]]
    _loaded_from_persistence: bool = False

    def __init__(
        self,
        data: pd.DataFrame,
        by: Union[str, List[str]],
        x: str,
        y: str,
        tile_size: int = DEFAULT_TILE_SIZE,
        importance: Optional[str] = None,
        importance_aggregation: AggregationMethod = 'mean',
        hierarchical: bool = False,
        zoom_range: Union[
            Tuple[float, float],
            List[Tuple[float, float]],
            Dict[str, Tuple[float, float]],
        ] = DEFAULT_ZOOM_RANGE,
        font: Union[Font, List[Font], Dict[str, Font]] = DEFAULT_FONT_FACE,
        size: Union[Auto, int, List[int], Dict[str, int]] = 'auto',
        color: Union[Auto, Color, List[Color], Dict[str, Color]] = 'auto',
        background: Color = 'white',
        bbox_percentile_range: Tuple[float, float] = (5, 95),
        max_labels_per_tile: int = DEFAULT_MAX_LABELS_PER_TILE,
        scale_function: LabelScaleFunction = 'constant',
        positioning: LabelPositioning = 'highest_density',
        exclude: List[str] = [],
        target_aspect_ratio: Optional[float] = None,
        max_lines: Optional[int] = None,
        verbosity: LogLevel = 'warning',
    ):
        """
        Initialize the Labeler with data and configuration.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with x, y coordinates and categorical columns.
        by : str or list of str
            Column name(s) used for labeling points. The referenced columns must
            contain either string or categorical values and are treated as
            categorical internally such that each category marks a group of
            points to be labeled as the category.

            To display individual point labels (where each row gets its own label),
            append an exclamation mark to the column name. For instance, `"city!"`
            would indicate that each value in this column should be treated as a
            unique label rather than grouping identical values together.

            Note: Currently only one column can be marked with an exclamation mark.
            If multiple columns are marked, only the first one is treated as
            containing point labels.
        x : str
            Name of the x-coordinate column.
        y : str
            Name of the y-coordinate column.
        tile_size : int, default=DEFAULT_TILE_SIZE
            Size of the tiles used for label placement in pixels. This determines
            the granularity of label density control and affects how labels are
            displayed at different zoom levels.
        importance : str, optional
            Column name containing importance values. These values determine which
            labels are prioritized when there are conflicts.
        importance_aggregation : {'min', 'mean', 'median', 'max', 'sum'}, default='mean'
            Method used to aggregate importance values when multiple points share
            the same label. This affects how label importance is calculated for
            groups of points.
        hierarchical : bool, default=False
            If True, the label types specified by `by` are expected to be
            hierarchical, which will affect the priority sorting of labels such
            that labels with a lower hierarchical index are displayed first.
        zoom_range : tuple of floats or list of tuple of floats or dict of tuple of floats, default=DEFAULT_ZOOM_RANGE
            The range at which labels of a specific type (as specified with `by`)
            are allowed to appear. The zoom range is a tuple of zoom levels,
            where `zoom_scale == 2 ** zoom_level`. Default is (-∞, ∞) for all labels.
        font : Font or list of Font or dict of Font, default=DEFAULT_FONT_FACE
            Font object(s) for text measurement. Can be specified as:
            - Single Font: Applied to all label types
            - List of Fonts: One font per label type in `by`
            - Dict: Maps label types or specific labels to fonts
        size : int or list of int or dict of int or 'auto', default='auto'
            Font size(s) for label text. Can be specified as:
            - 'auto': Automatically assign sizes (hierarchical if hierarchical=True)
            - Single int: Uniform size for all labels
            - List of ints: One size per label type in `by`
            - Dict: Maps label types or specific labels to sizes
        color : color or list of color or dict of color or 'auto', default='auto'
            Color specification for labels. Can be:
            - 'auto': Automatically choose based on background
            - str: Named color or hex code
            - tuple: RGB(A) values
            - list: Different colors for different hierarchy levels
            - dict: Mapping of label types or specific labels to colors
        background : color, default='white'
            Background color. Used for determining label colors when color='auto'.
        bbox_percentile_range : tuple of float, default=(5, 95)
            Range of percentiles to include when calculating the bounding box
            of points for label placement. This helps exclude outliers when
            determining where to place labels.
        max_labels_per_tile : int, default=DEFAULT_MAX_LABELS_PER_TILE
            Maximum number of labels per tile. Controls label density by limiting
            how many labels can appear in a given region. Set to 0 for unlimited.
        scale_function : {'asinh', 'constant'}, default='constant'
            Label zoom scale function for zoom level calculations:
            - 'asinh': Scales labels by the inverse hyperbolic sine, initially
              increasing linearly but quickly plateauing
            - 'constant': No scaling with zoom, labels maintain the same size
        positioning : {'highest_density', 'center_of_mass', 'largest_cluster'}, default='highest_density'
            Method used to determine the position of each label:
            - 'highest_density': Position label at the highest density point
            - 'center_of_mass': Position label at the center of mass of all points
            - 'largest_cluster': Position label at the center of the largest cluster
        exclude : list of str, default=[]
            Specifies which labels should be excluded. Can contain:
            - Column names (e.g., `"country"`) to exclude an entire category
            - Column-value pairs (e.g., `"country:USA"`) to exclude specific labels
        target_aspect_ratio : float, optional
            If not `None`, labels will potentially receive line breaks such that
            their bounding box is as close to the specified aspect ratio as
            possible. The aspect ratio is width/height.
        max_lines : int, optional
            Specify the maximum number of lines a label should be broken into if
            `target_aspect_ratio` is not `None`.
        verbosity : {'debug', 'info', 'warning', 'error', 'critical'}, default='warning'
            Controls the level of logging information displayed during label placement
            computation.
        """
        self._data = data
        self._x = x
        self._y = y
        self._hierarchical = hierarchical
        self._importance = importance
        self._importance_aggregation = importance_aggregation
        self._bbox_percentile_range = bbox_percentile_range
        self._tile_size = tile_size
        self._max_labels_per_tile = max_labels_per_tile
        self._scale_function = scale_function
        self._background = background
        self._target_aspect_ratio = target_aspect_ratio
        self._max_lines = max_lines
        self._verbosity = verbosity

        # Process 'by' parameter to handle special point-label marker
        by = by if isinstance(by, list) else [by]
        self._by = [b[:-1] if b.endswith('!') else b for b in by]
        self._point_label_columns = [b[:-1] for b in by if b.endswith('!')]

        labels = get_unique_labels(self._data, self._by)

        configure_logging(self._verbosity)

        # Validate only one point label column
        if len(self._point_label_columns) > 1:
            logger.warning(
                f'Multiple point label columns specified: {self._point_label_columns}. '
                f'Only the first one ({self._point_label_columns[0]}) will be used as a point label.'
            )
            self._point_label_columns = [self._point_label_columns[0]]

        self._exclude = map_binary_list_property(
            name='exclude',
            label_types=self._by,
            labels=labels,
            value=exclude,
        )

        self._validate_data()

        self._x_min = cast(float, self._data[x].min())
        self._x_max = cast(float, self._data[x].max())
        self._y_min = cast(float, self._data[y].min())
        self._y_max = cast(float, self._data[y].max())
        self._x_extent = self._x_max - self._x_min
        self._y_extent = self._y_max - self._y_min

        # Assign custom defaults
        if self._hierarchical:
            self._size = {by: max(16 - 2 * i, 8) for i, by in enumerate(self._by)}

        self._font = map_property(
            name='font',
            label_types=self._by,
            labels=labels,
            value=font,
            default_value=DEFAULT_FONT_FACE,
        )

        self._size = map_property(
            name='size',
            label_types=self._by,
            labels=labels,
            value=size,
            default_value=DEFAULT_FONT_SIZE,
            current_value=getattr(self, '_size', None),
        )

        self._color = map_property(
            name='color',
            label_types=self._by,
            labels=labels,
            value=color,
            default_value=self._default_color,
            current_value=getattr(self, '_color', None),
            value_transform=to_hex,
        )

        self._zoom_range = map_property(
            name='zoom ranges',
            label_types=self._by,
            labels=labels,
            value=zoom_range,
            default_value=DEFAULT_ZOOM_RANGE,
        )
        self._is_default_zoom_ranges = is_default_zoom_ranges(self._zoom_range)

        self._text = {by: Text(font) for by, font in self._font.items()}

        self._positioning = positioning

    @property
    def _default_color(self) -> str:
        background_luminance = (
            1.0 if self._background is None else calculate_luminance(self._background)
        )
        return 'black' if background_luminance > 0.5 else 'white'

    @property
    def computed(self) -> bool:
        """
        Get whether labels have been computed or not

        Returns
        -------
        bool
            If `True` the labels have been computed
        """
        return self._labels is not None

    @property
    def loaded_from_persistence(self) -> bool:
        """
        Get whether this is a restored instance.

        Returns
        -------
        bool
            If `True` the labels have been restored from files.
        """
        return self._loaded_from_persistence

    @property
    def background(self) -> Color:
        """
        Get the current background color.

        Returns
        -------
        Color
            The background color
        """
        return self._background

    @background.setter
    def background(self, background: Color):
        """
        Set new background color.

        Parameters
        ----------
        background : Color
            New fbackground color. Used for determining colors.
        """
        self._background = background

    @property
    def color(self) -> Dict[str, str]:
        """
        Get the current font color mapping.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping each label type and specific label to its color.
            This is an expanded version of what was provided during initialization,
            with colors assigned to all label types according to the mapping rules.
        """
        return self._color

    @color.setter
    def color(self, color: Union[Auto, Color, List[Color], Dict[str, Color]]) -> None:
        """
        Set new font colors and update labels if they exist.

        Parameters
        ----------
        color : Union[Auto, Color, List[Color], Dict[str, Color]]
            New color specification for labels. Can be:
            - str: Named color or hex code
            - tuple: RGB(A) values
            - list: Different colors for different hierarchy levels
            - dict: Mapping of label types to colors
        """

        self._color = map_property(
            name='font color',
            label_types=self._by,
            labels=get_unique_labels(self._data, self._by),
            value=color,
            default_value=self._default_color,
            current_value=getattr(self, '_color', None),
            value_transform=to_hex,
        )

        # Update processed labels if they exist
        if self._labels is not None:
            # Create a temporary series to hold the new colors
            new_colors = pd.Series(index=self._labels.index, dtype='object')

            # First, set colors based on label_type (lower priority)
            for label_type in self._by:
                mask = self._labels['label_type'] == label_type
                new_colors[mask] = self._color[label_type]

            # Then override with specific label colors (higher priority)
            for key, color in self._color.items():
                if ':' in key:  # This is a specific label color
                    label_type, label_value = key.split(':', 1)
                    mask = (self._labels['label_type'] == label_type) & (
                        self._labels['label'] == label_value
                    )
                    new_colors[mask] = color

            # Update the 'font_color' column
            self._labels['font_color'] = new_colors

    @property
    def font(self) -> Dict[str, Font]:
        """
        Get the current font face mapping.

        Returns
        -------
        Dict[str, Font]
            A dictionary mapping each label type and specific label to its font face.
            This is an expanded version of what was provided during initialization,
            with font faces assigned to all label types according to the mapping rules.

        Note
        ----
        This property is read-only after labels have been computed as changing
        font faces may affect spatial placement.
        """
        return self._font

    @font.setter
    def font(self, font: Union[Font, List[Font], Dict[str, Font]]):
        """
        Set new font faces.

        Parameters
        ----------
        font : Union[Font, List[Font], Dict[str, Font]]
            New font face specification for labels.

        Raises
        ------
        ValueError
            If trying to update font faces after labels have been computed.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update font faces after having computed labels. Spatial properties like font face '
                'may affect label placement and require reprocessing. Use reset() first or '
                'create a new LabelPlacement instance with the desired font faces.'
            )

        self._font = map_property(
            name='font',
            label_types=self._by,
            labels=get_unique_labels(self._data, self._by),
            value=font,
            default_value=DEFAULT_FONT_FACE,
            current_value=getattr(self, '_font', None),
        )

        # Update text objects with the new fonts
        self._text = {by: Text(font) for by, font in self._font.items()}

    @property
    def size(self) -> Dict[str, int]:
        """
        Get the current font size mapping.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping each label type and specific label to its size.
            This is an expanded version of what was provided during initialization,
            with sizes assigned to all label types according to the mapping rules.

        Note
        ----
        This property is read-only after labels have been computed as changing
        font sizes may affect spatial placement.
        """
        return self._size

    @size.setter
    def size(self, size: Union[Auto, int, List[int], Dict[str, int]] = 'auto'):
        """
        Set new font size.

        Parameters
        ----------
        size : Union[Auto, int, List[int], Dict[str, int]]
            New font size specification for labels.

        Raises
        ------
        ValueError
            If trying to update font sizes after labels have been computed.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update font sizes after having computed labels. Spatial properties like size '
                'affect label placement and require reprocessing. Create a new '
                'LabelPlacement instance with the desired sizes instead.'
            )

        self._size = map_property(
            name='font size',
            label_types=self._by,
            labels=get_unique_labels(self._data, self._by),
            value=size,
            default_value=DEFAULT_FONT_SIZE,
            current_value=getattr(self, '_size', None),
        )

    @property
    def zoom_range(self) -> Dict[str, Tuple[np.float64, np.float64]]:
        """
        Get the current zoom range mapping.

        Returns
        -------
        Dict[str, Tuple[np.float64, np.float64]]
            A dictionary mapping each label type and specific label to its zoom range.

        Note
        ----
        This property is read-only after labels have been computed as changing
        zoom ranges affects spatial placement.
        """
        return self._zoom_range

    @zoom_range.setter
    def zoom_range(
        self,
        zoom_range: Union[
            Tuple[float, float],
            List[Tuple[float, float]],
            Dict[str, Tuple[float, float]],
        ],
    ):
        """
        Set new zoom ranges.

        Parameters
        ----------
        zoom_range : Union[Tuple[float, float], List[Tuple[float, float]], Dict[str, Tuple[float, float]]]
            New zoom range specification for labels.

        Raises
        ------
        ValueError
            If trying to update zoom ranges after labels have been computed.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update zoom ranges after having computed labels. Spatial properties like zoom range '
                'affect label placement and require reprocessing. Use reset() first or '
                'create a new LabelPlacement instance with the desired zoom ranges.'
            )

        self._zoom_range = map_property(
            name='zoom ranges',
            label_types=self._by,
            labels=get_unique_labels(self._data, self._by),
            value=zoom_range,
            default_value=DEFAULT_ZOOM_RANGE,
        )
        self._is_default_zoom_ranges = is_default_zoom_ranges(self._zoom_range)

    @property
    def exclude(self) -> List[str]:
        """
        Get the current exclude mapping.

        Returns
        -------
        List[str]
            A list of label types and specific labels to be excluded.
        """
        return list(self._exclude)

    @exclude.setter
    def exclude(self, exclude: List[str]):
        """
        Set new exclude values.

        Parameters
        ----------
        exclude : list[str]
            New exclude specification for labels.

        Raises
        ------
        ValueError
            If trying to update exclude settings after labels have been computed.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update exclude settings after having computed labels. Changing which labels '
                'are excluded affects label placement and requires reprocessing. Use reset() first or '
                'create a new LabelPlacement instance with the desired exclude settings.'
            )

        self._exclude = map_binary_list_property(
            name='exclude',
            label_types=self._by,
            labels=get_unique_labels(self._data, self._by),
            value=exclude,
        )

    @property
    def data(self) -> pd.DataFrame:
        """Get the input data."""
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame):
        """
        Set new input data.

        Parameters
        ----------
        data : pd.DataFrame
            New data for label placement.

        Raises
        ------
        ValueError
            If trying to update data after labels have been computed or if the new data
            would result in different unique labels.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update data after having computed labels. '
                'Use reset() first or create a new instance.'
            )

        # Check if new data would result in the same unique labels
        current_labels = set(get_unique_labels(self._data, self._by))
        new_labels = set(get_unique_labels(data, self._by))

        if current_labels != new_labels:
            raise ValueError(
                'Cannot update data as it would result in different unique labels. '
                'Use clone() or create a new instance instead.'
            )

        self._data = data
        self._validate_data()

        # Update data-dependent attributes
        self._x_min = cast(float, self._data[self._x].min())
        self._x_max = cast(float, self._data[self._x].max())
        self._y_min = cast(float, self._data[self._y].min())
        self._y_max = cast(float, self._data[self._y].max())
        self._x_extent = self._x_max - self._x_min
        self._y_extent = self._y_max - self._y_min

    @property
    def x(self) -> str:
        """Get the name of the x-coordinate column."""
        return self._x

    @x.setter
    def x(self, x: str):
        """
        Set new x-coordinate column name.

        Parameters
        ----------
        x : str
            New x-coordinate column name.

        Raises
        ------
        ValueError
            If trying to update after labels have been computed or if column doesn't exist.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update x-coordinate column after having computed labels. '
                'Use reset() first or create a new instance.'
            )

        if x not in self._data.columns:
            raise ValueError(f"Column '{x}' not found in data.")

        self._x = x

        # Update coordinate extents
        self._x_min = cast(float, self._data[self._x].min())
        self._x_max = cast(float, self._data[self._x].max())
        self._x_extent = self._x_max - self._x_min

    @property
    def y(self) -> str:
        """Get the name of the y-coordinate column."""
        return self._y

    @y.setter
    def y(self, y: str):
        """
        Set new y-coordinate column name.

        Parameters
        ----------
        y : str
            New y-coordinate column name.

        Raises
        ------
        ValueError
            If trying to update after labels have been computed or if column doesn't exist.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update y-coordinate column after having computed labels. '
                'Use reset() first or create a new instance.'
            )

        if y not in self._data.columns:
            raise ValueError(f"Column '{y}' not found in data.")

        self._y = y

        # Update coordinate extents
        self._y_min = cast(float, self._data[self._y].min())
        self._y_max = cast(float, self._data[self._y].max())
        self._y_extent = self._y_max - self._y_min

    @property
    def by(self) -> List[str]:
        """Get the column name(s) defining the label hierarchy."""
        return self._by

    @by.setter
    def by(self, by: Union[str, List[str]]):
        """
        Set new label hierarchy column(s).

        Parameters
        ----------
        by : Union[str, List[str]]
            New column name(s) for label hierarchy.

        Raises
        ------
        ValueError
            If trying to update after labels have been computed or if columns don't exist
            or if the new 'by' columns would result in different unique labels.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update label columns after having computed labels. '
                'Use reset() first or create a new instance.'
            )

        by_list = by if isinstance(by, list) else [by]
        missing_cols = [col for col in by_list if col not in self._data.columns]
        if missing_cols:
            raise ValueError(f'Columns not found in data: {missing_cols}')

        # Check if new 'by' columns would result in the same unique labels
        current_labels = set(get_unique_labels(self._data, self._by))
        new_labels = set(get_unique_labels(self._data, by_list))

        if current_labels != new_labels:
            raise ValueError(
                'Cannot update "by" columns as it would result in different unique labels. '
                'Use clone() or create a new instance instead.'
            )

        self._by = by_list
        self._validate_data()

    @property
    def hierarchical(self) -> bool:
        """Get whether the labels are hierarchical."""
        return self._hierarchical

    @hierarchical.setter
    def hierarchical(self, hierarchical: bool):
        """
        Set whether the labels are hierarchical.

        Parameters
        ----------
        hierarchical : bool
            Whether the labels are hierarchical.

        Raises
        ------
        ValueError
            If trying to update after labels have been computed.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update hierarchical setting after having computed labels. '
                'Use reset() first or create a new instance.'
            )

        self._hierarchical = hierarchical
        self._validate_data()

    @property
    def importance(self) -> Optional[str]:
        """Get the name of the importance column."""
        return self._importance

    @importance.setter
    def importance(self, importance: Optional[str]):
        """
        Set new importance column name.

        Parameters
        ----------
        importance : Optional[str]
            New importance column name.

        Raises
        ------
        ValueError
            If trying to update after labels have been computed or if column doesn't exist.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update importance column after having computed labels. '
                'Use reset() first or create a new instance.'
            )

        if importance is not None and importance not in self._data.columns:
            raise ValueError(f"Column '{importance}' not found in data.")

        self._importance = importance

    @property
    def importance_aggregation(self) -> AggregationMethod:
        """Get the importance aggregation method."""
        return self._importance_aggregation

    @importance_aggregation.setter
    def importance_aggregation(self, aggregation: AggregationMethod):
        """
        Set new importance aggregation method.

        Parameters
        ----------
        aggregation : AggregationMethod
            New importance aggregation method. Can be one of `"min"`, `"mean"`, `"median"`, `"max"`, `"sum"`

        Raises
        ------
        ValueError
            If trying to update after labels have been computed or if column doesn't exist.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update importance aggregation methid after having '
                'computed labels. Use reset() first or create a new instance.'
            )

        self._importance_aggregation = aggregation

    @property
    def bbox_percentile_range(self) -> Tuple[float, float]:
        """Get the percentile range for bounding box calculation."""
        return self._bbox_percentile_range

    @bbox_percentile_range.setter
    def bbox_percentile_range(self, bbox_percentile_range: Tuple[float, float]):
        """
        Set new percentile range for bounding box calculation.

        Parameters
        ----------
        bbox_percentile_range : Tuple[float, float]
            New percentile range for bounding box calculation.

        Raises
        ------
        ValueError
            If trying to update after labels have been computed.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update bbox percentile range after having computed labels. '
                'Use reset() first or create a new instance.'
            )

        self._bbox_percentile_range = bbox_percentile_range

    @property
    def tile_size(self) -> int:
        """Get the tile size."""
        return self._tile_size

    @tile_size.setter
    def tile_size(self, tile_size: int):
        """
        Set new tile size.

        Parameters
        ----------
        tile_size : int
            New tile size.

        Raises
        ------
        ValueError
            If trying to update after labels have been computed.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update tile size after having computed labels. '
                'Use reset() first or create a new instance.'
            )

        self._tile_size = tile_size

    @property
    def max_labels_per_tile(self) -> int:
        """Get the maximum number of labels per tile."""
        return self._max_labels_per_tile

    @max_labels_per_tile.setter
    def max_labels_per_tile(self, max_labels_per_tile: int):
        """
        Set new maximum number of labels per tile.

        Parameters
        ----------
        max_labels_per_tile : int
            New maximum number of labels per tile.

        Raises
        ------
        ValueError
            If trying to update after labels have been computed.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update maximum labels per tile after having computed labels. '
                'Use reset() first or create a new instance.'
            )

        self._max_labels_per_tile = max_labels_per_tile

    @property
    def scale_function(self) -> LabelScaleFunction:
        """Get the label zoom scale function."""
        return self._scale_function

    @scale_function.setter
    def scale_function(self, scale_function: LabelScaleFunction):
        """
        Set new label zoom scale function.

        Parameters
        ----------
        scale_function : LabelScaleFunction
            New label zoom scale function.

        Raises
        ------
        ValueError
            If trying to update after labels have been computed.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update scale function after having computed labels. '
                'Use reset() first or create a new instance.'
            )

        self._scale_function = scale_function

    @property
    def positioning(self) -> LabelPositioning:
        """Get the label positioning method."""
        return self._positioning

    @positioning.setter
    def positioning(self, positioning: LabelPositioning):
        """
        Set new label positioning method.

        Parameters
        ----------
        positioning : LabelPositioning
            New label positioning method.

        Raises
        ------
        ValueError
            If trying to update after labels have been computed.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update positioning method after having computed labels. '
                'Use reset() first or create a new instance.'
            )

        if positioning == 'largest_cluster':
            check_label_extras_dependencies()

        self._positioning = positioning

    @property
    def target_aspect_ratio(self) -> Optional[float]:
        """Get the target aspect ratio for line break optimization."""
        return self._target_aspect_ratio

    @target_aspect_ratio.setter
    def target_aspect_ratio(self, target_aspect_ratio: Optional[float]):
        """
        Set the target aspect ratio for line break optimization.

        Parameters
        ----------
        target_aspect_ratio : float or None
            Target aspect ratio (width/height), or None to disable line break optimization

        Raises
        ------
        ValueError
            If trying to update after labels have been computed.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update target aspect ratio after having computed labels. '
                'Use reset() first or create a new instance.'
            )

        self._target_aspect_ratio = target_aspect_ratio

    @property
    def max_lines(self) -> Optional[int]:
        """Get the maximum number of lines for line break optimization."""
        return self._max_lines

    @max_lines.setter
    def max_lines(self, max_lines: Optional[int]):
        """
        Set the maximum number of lines for line break optimization.

        Parameters
        ----------
        max_lines : int or None
            Maximum number of lines, or None to disable line break optimization

        Raises
        ------
        ValueError
            If trying to update after labels have been computed.
        """
        if self._labels is not None:
            raise ValueError(
                'Cannot update maximum lines after having computed labels. '
                'Use reset() first or create a new instance.'
            )

        self._max_lines = max_lines

    @property
    def verbosity(self) -> LogLevel:
        """
        Get the current log level.

        Returns
        -------
        LogLevel
            The current log level
        """
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity: LogLevel) -> None:
        """
        Set new log level.

        Parameters
        ----------
        verbosity : LogLevel
            New log level
        """

        self._verbosity = verbosity
        configure_logging(self._verbosity)

    # Read-only properties after computation
    @property
    def labels(self) -> Optional[pd.DataFrame]:
        """Get the computed labels, if available."""
        return self._labels

    @property
    def tiles(self) -> Optional[Dict[str, List[int]]]:
        """Get the tile mapping, if available."""
        return self._tiles

    def reset(self) -> None:
        """
        Reset the computed labels, allowing spatial properties to be modified
        before recomputing labels.

        Note: This method clears existing labels and tiles to allow spatial
        properties to be changed. Call compute() again after modifying
        properties.
        """
        self._labels = None
        self._tiles = None

    def clone(self, **kwargs):
        """
        Create a new LabelPlacement instance with the same configuration,
        optionally overriding specific parameters.

        Parameters
        ----------
        **kwargs
            Any parameters to override from the current instance

        Returns
        -------
        LabelPlacement
            A new instance with the specified configuration
        """
        # Get current parameters
        params = {
            'data': self._data,
            'by': self._by,
            'x': self._x,
            'y': self._y,
            'tile_size': self._tile_size,
            'importance': self._importance,
            'hierarchical': self._hierarchical,
            'zoom_range': self._zoom_range,
            'font': self._font,
            'size': self._size,
            'color': self._color,
            'background': self._background,
            'bbox_percentile_range': self._bbox_percentile_range,
            'max_labels_per_tile': self._max_labels_per_tile,
            'scale_function': self._scale_function,
            'positioning': self._positioning,
            'exclude': list(self._exclude),
            'target_aspect_ratio': self._target_aspect_ratio,
            'max_lines': self._max_lines,
        }

        # Override with provided kwargs
        params.update(kwargs)

        return LabelPlacement(**params)

    def compute(
        self,
        show_progress=False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> pd.DataFrame:
        """
        Compute the labels with full collision detection and density control.
        Main method that combines all preprocessing and placement steps.

        Returns
        -------
        pandas.DataFrame
            Computed labels ready for rendering
        """
        # Check for extra dependencies
        if show_progress or self._positioning == 'largest_cluster':
            check_label_extras_dependencies()

        import time

        if show_progress:
            progress_bar = tqdm(
                total=self._get_total_num_operations(),
                desc='Compute labels',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [Total: {elapsed}]',
                miniters=1,
            )
        else:
            progress_bar = None

        # Step 1: Preprocess labels to get basic information
        t0 = time.perf_counter()
        self._labels = self._create_labels(progress_bar=progress_bar)
        t1 = time.perf_counter()
        logger.info(f'Creating {len(self._labels)} labels took {t1 - t0} seconds')

        if self._labels.empty:
            return self._labels

        # Step 2: Calculate initial zoom levels
        t0 = time.perf_counter()
        self._labels = self._compute_zoom_levels(
            self._labels, progress_bar=progress_bar
        )
        t1 = time.perf_counter()
        logger.info(
            f'Computing zoom levels for {len(self._labels)} labels took {t1 - t0} seconds'
        )

        # Step 3: Build spatial index
        t0 = time.perf_counter()
        spatial_index = self._build_spatial_index(
            self._labels, progress_bar=progress_bar
        )
        t1 = time.perf_counter()
        logger.info(
            f'Building the rtree for {len(self._labels)} labels took {t1 - t0} seconds'
        )

        # Step 4: Resolve label collisions
        t0 = time.perf_counter()
        self._labels = self._resolve_collisions(
            self._labels,
            spatial_index,
            progress_bar=progress_bar,
            chunk_size=chunk_size,
        )
        t1 = time.perf_counter()
        logger.info(
            f'Resolving collisions for {len(self._labels)} labels took {t1 - t0} seconds'
        )

        # Step 5: Apply density control if needed
        t0 = time.perf_counter()
        self._labels = self._tile_labels(
            self._labels,
            max(1, self._max_labels_per_tile),
            min_zoom_extent=1,
            max_zoom_level=MAX_ZOOM_LEVEL,
            progress_bar=progress_bar,
        )
        t1 = time.perf_counter()
        logger.info(f'Tiling {len(self._labels)} labels took {t1 - t0} seconds')

        # Step 6: Compute zoom extent and fade in/out
        t0 = time.perf_counter()
        self._labels = self._compute_zoom_extent_fade_in_out(self._labels)
        t1 = time.perf_counter()
        logger.info(
            f'Computing the zoom extent fading for {len(self._labels)} labels took {t1 - t0} seconds'
        )

        if progress_bar:
            progress_bar.set_description(f'Process labels')

        # Return processed data
        return self._labels

    def to_parquet(self, path: str, format: PersistenceFormat = 'parquet') -> None:
        """
        Export label placement data to storage.

        Parameters
        ----------
        path : str
            Path where the data will be stored. For parquet format, this should be a directory.
            For arrow_ipc format, this should be a file path.
        format : str, default="parquet"
            Format to use for persistence. Options are "parquet" or "arrow_ipc".
        """
        from .persistence import to_parquet

        return to_parquet(self, path, format)

    @staticmethod
    def from_parquet(
        path: str, format: Optional[PersistenceFormat] = None
    ) -> 'LabelPlacement':
        """
        Load label placement data from storage.

        Parameters
        ----------
        path : str
            Path where the data is stored. For parquet format, this should be a directory.
            For `"arrow_ipc"` format, this should be a file path.
        format : "parquet" or "arrow_ipc", optional
            Format to use for loading. Options are "parquet" or "arrow_ipc".
            If None, will be determined from the path.

        Returns
        -------
        LabelPlacement
            Loaded label placement object
        """
        from .persistence import from_parquet

        return from_parquet(path, format)

    def get_labels_from_tiles(self, tile_ids: List[str]):
        """
        Get labels from data tiles.

        Parameters
        ----------
        tile_ids : list of str
            Tile IDs

        Returns
        -------
        pandas.DataFrame
            Labels from the data tiles
        """
        if self._labels is None:
            return pd.DataFrame()
        label_idxs = self._get_label_idxs_from_tiles(tile_ids)
        return self._labels.loc[label_idxs]

    def _validate_data(self):
        """Validate the input data and check for hierarchy consistency."""
        # Check if all columns exist
        required_cols = [self._x, self._y] + self._by
        if self._importance:
            required_cols.append(self._importance)

        missing_cols = [col for col in required_cols if col not in self._data.columns]
        if missing_cols:
            raise ValueError(f'Missing columns in data: {missing_cols}')

        # Check hierarchy structure - each label in a lower hierarchy must be a subset of a higher hierarchy
        if len(self._by) > 1 and self._hierarchical:
            # Track if we've seen a point label column in the hierarchy
            seen_point_label = False

            for i in range(1, len(self._by)):
                higher_level = self._by[i - 1]
                lower_level = self._by[i]

                # If we've already seen a point label column, all subsequent columns must also be point labels
                if seen_point_label and lower_level not in self._point_label_columns:
                    raise ValueError(
                        f"Hierarchy violation: '{lower_level}' follows a point label column but is not a point label itself. "
                        f'All columns after a point label column must also be point labels.'
                    )

                # Skip hierarchy validation if the current level is a point label column
                if lower_level in self._point_label_columns:
                    seen_point_label = True
                    continue

                # Update seen_point_label if higher level is a point label
                if higher_level in self._point_label_columns:
                    seen_point_label = True
                    # No need to validate hierarchy as we'll continue to the next iteration
                    continue

                # Create a filtered dataframe excluding points based on exclude list
                filtered_data = self._data.copy()

                # Filter out rows where higher_level values are excluded
                if len(self._exclude):
                    # Filter rows where the higher level category is excluded
                    higher_level_mask = filtered_data[higher_level].apply(
                        lambda x: f'{higher_level}:{x}' not in self._exclude
                    )

                    # Filter rows where the lower level category is excluded
                    lower_level_mask = filtered_data[lower_level].apply(
                        lambda x: f'{lower_level}:{x}' not in self._exclude
                    )

                    # Apply both filters
                    filtered_data = filtered_data[higher_level_mask & lower_level_mask]

                # Group by the lower level and check if each group has exactly one higher level value
                # But only for non-excluded values
                grp = filtered_data.groupby(lower_level, observed=True)[
                    higher_level
                ].nunique()
                invalid_groups = grp[grp > 1].index.tolist()

                if invalid_groups:
                    raise ValueError(
                        f"Hierarchy violation: The following values in '{lower_level}' belong to multiple parents in '{higher_level}': {invalid_groups[:5]}"
                        + ('...' if len(invalid_groups) > 5 else '')
                    )

    def _compute_label_hash(self, label_name: str, point_indices):
        """Create a deterministic hash for a label based on name and minimum point index."""
        min_index = np.min(point_indices)
        hash_input = f'{label_name}_{min_index}'
        return int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 10**10

    def _compute_bbox(
        self, points: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute the bounding box and label position given a set of points.

        Parameters
        ----------
        points : numpy.ndarray
            Array of [x, y] points

        Returns
        -------
        bbox : numpy.ndarray
            Bounding box as [min_x, min_y, max_x, max_y]
        position : numpy.ndarray
            Label center as [x, y]
        """
        if points.size == 0:
            return np.array([0, 0, 0, 0]), np.array([0, 0])

        # Compute bounding box
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        bbox: npt.NDArray[np.float64] = np.array([min_x, min_y, max_x, max_y])

        if self._positioning == 'highest_density':
            position = compute_highest_density_point(points)
            return bbox, position

        if self._positioning == 'largest_cluster':
            check_label_extras_dependencies()
            points = compute_largest_cluster(points)

        if len(points) < 3:
            # Not enough points for convex hull, use the points themselves
            # Simple centroid for 1-2 points
            position = np.mean(points, axis=0)
        else:
            # Compute convex hull
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]

                # Calculate center of mass using Shoelace formula
                position = compute_center_of_mass(hull_points)
            except QhullError:
                # Fall back to the mean
                position = np.mean(points, axis=0)

        return bbox, position

    def _compute_zoom_levels(
        self, labels: pd.DataFrame, progress_bar: Optional[tqdm] = None
    ) -> pd.DataFrame:
        """
        Calculate initial zoom levels for labels based on their dimensions.

        Parameters
        ----------
        labels : pandas.DataFrame
            Label data with coordinates and dimensions

        Returns
        -------
        pandas.DataFrame
            Updated labels with zoom levels added
        """

        if progress_bar:
            progress_bar.set_description('Computing zoom levels')

        n = len(labels)

        x_scale = create_linear_scale([0, self._tile_size], [0, self._x_extent])
        y_scale = create_linear_scale([0, self._tile_size], [0, self._y_extent])

        zoom_in = np.float64(ZOOM_IN_PX / self._tile_size)

        # Pre-allocate numpy arrays for our calculations
        widths = np.zeros(n).astype(np.float64)
        heights = np.zeros(n).astype(np.float64)
        zoom_ins = np.ones(n).astype(np.float64) * zoom_in
        zoom_outs = np.ones(n).astype(np.float64) * math.inf
        min_zoom_ins = np.zeros(n).astype(np.float64)
        max_zoom_outs = np.ones(n).astype(np.float64) * math.inf

        # Get all the data we need as numpy arrays for faster processing
        font_sizes = cast(npt.NDArray[np.uint8], labels['font_size'].values)
        label_types = cast(list[str], labels['label_type'].values)
        label_texts = cast(list[str], labels['label'].values)
        bbox_widths = cast(npt.NDArray[np.float64], labels['bbox_width'].values)
        bbox_heights = cast(npt.NDArray[np.float64], labels['bbox_height'].values)

        for i in range(n):
            # Get the specific zoom range for this label type
            min_zoom_level, max_zoom_level = self._zoom_range.get(
                f'{label_types[i]}:{label_texts[i]}',
                self._zoom_range.get(label_types[i], DEFAULT_ZOOM_RANGE),
            )

            min_zoom_ins[i] = 2**min_zoom_level
            max_zoom_outs[i] = 2**max_zoom_level

            text = self._text.get(
                f'{label_types[i]}:{remove_line_breaks(label_texts[i])}',
                self._text[label_types[i]],
            )
            text_width, text_height = text.measure(label_texts[i], font_sizes[i])

            widths[i] = x_scale(text_width)
            heights[i] = y_scale(text_height)

            if progress_bar:
                progress_bar.update()

        # Apply minimum and maximum zoom constraints
        zoom_ins = np.maximum(zoom_ins, min_zoom_ins)
        zoom_outs = np.minimum(zoom_outs, max_zoom_outs)

        # Precompute the zoom out level components
        x_zoom_components = np.ones(n) * math.inf  # Initialize with infinity
        y_zoom_components = np.ones(n) * math.inf  # Initialize with infinity

        # Only calculate for non-zero dimensions
        non_zero_width_mask = bbox_widths > 0
        non_zero_height_mask = bbox_heights > 0

        x_zoom_components[non_zero_width_mask] = (
            self._x_extent / bbox_widths[non_zero_width_mask]
        ) * 2
        y_zoom_components[non_zero_height_mask] = (
            self._y_extent / bbox_heights[non_zero_height_mask]
        ) * 2

        # Calculate max of x and y components
        zoom_components_max = np.maximum(x_zoom_components, y_zoom_components)

        # Calculate final zoom levels
        zoom_outs = np.minimum(zoom_outs, np.maximum(zoom_components_max, zoom_in * 2))

        labels['width'] = widths
        labels['height'] = heights
        labels['zoom_in'] = zoom_ins
        labels['zoom_out'] = zoom_outs
        labels['min_zoom_in'] = min_zoom_ins
        labels['max_zoom_out'] = max_zoom_outs

        return labels

    def _build_spatial_index(
        self,
        labels,
        zoom_scale: np.float64 = np.float64(1.0),
        label_idxs: Optional[Union[List[int], np.ndarray]] = None,
        progress_bar: Optional[tqdm] = None,
    ):
        """
        Build spatial index for efficient collision detection.

        Parameters
        ----------
        labels : pandas.DataFrame
            Label data with coordinates and dimensions
        zoom_scale : float, default=1.0
            Current zoom level
        label_idxs : list of int or numpy.ndarray, optional
            Specific point indices to include (optional)

        Returns
        -------
        Spatial index for collision detection
        """

        if progress_bar:
            progress_bar.set_description('Building spatial index')

        # Determine number of points to index
        if label_idxs is not None:
            n = len(label_idxs)
            get_idx = lambda i: label_idxs[i]
        else:
            n = len(labels)
            get_idx = lambda i: i

        indices = [get_idx(i) for i in range(n)]

        # Extract data for faster access
        center_xs = labels['x'].values[indices]
        center_ys = labels['y'].values[indices]
        half_widths = labels['width'].values[indices] / 2 / zoom_scale
        half_heights = labels['height'].values[indices] / 2 / zoom_scale

        # Calculate bbox
        min_xs = center_xs - half_widths
        min_ys = center_ys - half_heights
        max_xs = center_xs + half_widths
        max_ys = center_ys + half_heights

        builder = rtree.RTreeBuilder(num_items=n)
        builder.add(min_xs, min_ys, max_xs, max_ys)
        index = builder.finish()

        if progress_bar:
            progress_bar.update(n)

        return index

    def _resolve_zoom_out_collisions(self, labels: pd.DataFrame):
        """
        Resolve collisions between asinh-scaled labels when zooming out.

        Parameters
        ----------
        labels : pandas.DataFrame
            Label data with coordinates and dimensions

        Returns
        -------
        pandas.DataFrame
            Updated labels with collision-free zoom levels
        """
        center_xs = cast(npt.NDArray[np.float64], labels['x'].values)
        center_ys = cast(npt.NDArray[np.float64], labels['y'].values)
        half_widths = cast(npt.NDArray[np.float64], labels['width'].values) / 2.0
        half_heights = cast(npt.NDArray[np.float64], labels['height'].values) / 2.0
        zoom_ins = cast(npt.NDArray[np.float64], labels['zoom_in'].values)
        zoom_outs = cast(npt.NDArray[np.float64], labels['zoom_out'].values)

        # Resolve conflicts when zooming out (zoom scale < 1)
        label_idxs_zoom_in_smaller_one = np.where(zoom_ins < 1)[0]

        if len(label_idxs_zoom_in_smaller_one) > 0:
            min_zoom_in = np.min(zoom_ins)

            index_at_min_zoom_level = self._build_spatial_index(
                labels, min_zoom_in, label_idxs_zoom_in_smaller_one
            )

            # Pre-compute all bounding boxes for the second phase
            zoomed_half_widths = (
                half_widths[label_idxs_zoom_in_smaller_one] / min_zoom_in
            )
            zoomed_half_heights = (
                half_heights[label_idxs_zoom_in_smaller_one] / min_zoom_in
            )
            zoomed_min_xs = (
                center_xs[label_idxs_zoom_in_smaller_one] - zoomed_half_widths
            )
            zoomed_min_ys = (
                center_ys[label_idxs_zoom_in_smaller_one] - zoomed_half_heights
            )
            zoomed_max_xs = (
                center_xs[label_idxs_zoom_in_smaller_one] + zoomed_half_widths
            )
            zoomed_max_ys = (
                center_ys[label_idxs_zoom_in_smaller_one] + zoomed_half_heights
            )

            for j, i in enumerate(label_idxs_zoom_in_smaller_one):
                # Get colliding labels at zoom level min_zoom_in
                collisions = rtree.search(
                    index_at_min_zoom_level,
                    zoomed_min_xs[j],
                    zoomed_min_ys[j],
                    zoomed_max_xs[j],
                    zoomed_max_ys[j],
                ).to_numpy()
                # Map to original indices
                collisions = np.asarray(label_idxs_zoom_in_smaller_one[collisions])
                # Filter for higher priority
                collisions = collisions[collisions < i]

                if len(collisions) == 0:
                    continue

                center_x = center_xs[i]
                center_y = center_ys[i]
                half_width = zoomed_half_widths[j]  # Already scaled
                half_height = zoomed_half_heights[j]  # Already scaled

                # Vectorized resolution for the zoomed-out case
                colliding_center_xs = center_xs[collisions]
                colliding_center_ys = center_ys[collisions]
                colliding_half_widths = half_widths[collisions] / min_zoom_in
                colliding_half_heights = half_heights[collisions] / min_zoom_in

                # Vectorized calculations
                widths = half_width + colliding_half_widths
                heights = half_height + colliding_half_heights
                d_xs = np.abs(center_x - colliding_center_xs)
                d_ys = np.abs(center_y - colliding_center_ys)

                x_resolutions = np.divide(
                    widths, d_xs, out=np.full_like(d_xs, np.inf), where=d_xs > 0
                )
                y_resolutions = np.divide(
                    heights, d_ys, out=np.full_like(d_ys, np.inf), where=d_ys > 0
                )

                # Simple closed-form solution for this case
                resolution_zooms = np.minimum(x_resolutions, y_resolutions)

                # Apply min_zoom_in scaling
                new_zoom_ins = np.maximum(0, resolution_zooms * min_zoom_in)

                # Filter for collisions that require zoom adjustment
                valid_collisions = np.where(new_zoom_ins > zoom_ins[collisions])[0]

                if len(valid_collisions) > 0:
                    max_new_zoom_in = np.max(new_zoom_ins[valid_collisions])
                    zoom_ins[i] = max(zoom_ins[i], max_new_zoom_in)

        labels['zoom_in'] = zoom_ins
        labels['zoom_out'] = zoom_outs

        return labels

    def _resolve_collisions(
        self,
        labels: pd.DataFrame,
        spatial_index: Any,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress_bar: Optional[tqdm] = None,
    ):
        """
        Resolve collisions between labels by adjusting zoom levels.

        Parameters
        ----------
        labels : pandas.DataFrame
            Label data with coordinates and dimensions
        spatial_index : Spatial index for collision detection
        chunk_size : int, optional
            The size of labels that should be resolved simultaneously if possible
        progress_bar : tqdm.tqdm, optional
            Progress bar for tracking computation

        Returns
        -------
        pandas.DataFrame
            Updated labels with collision-free zoom levels
        """

        import time

        if progress_bar:
            progress_bar.set_description('Resolving collisions')

        center_xs = cast(npt.NDArray[np.float64], labels['x'].values)
        center_ys = cast(npt.NDArray[np.float64], labels['y'].values)
        half_widths = cast(npt.NDArray[np.float64], labels['width'].values) / 2.0
        half_heights = cast(npt.NDArray[np.float64], labels['height'].values) / 2.0
        zoom_ins = cast(npt.NDArray[np.float64], labels['zoom_in'].values)
        zoom_outs = cast(npt.NDArray[np.float64], labels['zoom_out'].values)
        min_zoom_ins = cast(npt.NDArray[np.float64], labels['min_zoom_in'].values)
        max_zoom_outs = cast(npt.NDArray[np.float64], labels['max_zoom_out'].values)

        n = len(labels)

        resolver = (
            resolve_static_zoom_in_collisions
            if self._scale_function == 'constant'
            else resolve_asinh_zoom_in_collisions
        )

        t1 = time.perf_counter()

        chunks: List[npt.NDArray[np.uint64]] = []
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunks.append(np.arange(i, end, dtype=np.uint64))

        # Pre-compute all bounding boxes
        min_xs = center_xs - half_widths
        min_ys = center_ys - half_heights
        max_xs = center_xs + half_widths
        max_ys = center_ys + half_heights

        for chunk in chunks:
            resolver(
                spatial_index=bytes(spatial_index),
                center_xs=center_xs,
                center_ys=center_ys,
                half_widths=half_widths,
                half_heights=half_heights,
                min_xs=min_xs,
                min_ys=min_ys,
                max_xs=max_xs,
                max_ys=max_ys,
                zoom_ins=zoom_ins,
                zoom_outs=zoom_outs,
                idxs=chunk,
                progress_bar=progress_bar,
            )

        # Ensure we're not violating zoom ranges
        zoom_ins = np.maximum(zoom_ins, min_zoom_ins)

        # Possibly increase the zoom out level to avoid that labels are only
        # shown for a very narrow zoom range
        zoom_outs = np.minimum(
            np.maximum(zoom_outs, zoom_ins * 2),
            max_zoom_outs,
        )

        # For cases where the zoom in is equal or larger than the zoon out we
        # set both to infinity to indicate that this label's collision can't be
        # resolved and that the label should not be shown
        mask = zoom_ins >= zoom_outs
        zoom_ins[mask] = np.float64(math.inf)
        zoom_outs[mask] = np.float64(math.inf)

        labels['zoom_in'] = zoom_ins
        labels['zoom_out'] = zoom_outs

        t2 = time.perf_counter()
        logger.info(f'Resolving zoom-in collisions took {t2 - t1} seconds')

        labels = self._resolve_zoom_out_collisions(labels)

        t3 = time.perf_counter()
        logger.info(f'Resolving zoom-out collisions took {t3 - t2} seconds')

        return labels

    def _get_tile_id(self, x: float, y: float, zoom_level: int):
        return get_tile_id(
            x=x,
            y=y,
            zoom_level=zoom_level,
            x_min=self._x_min,
            x_extent=self._x_extent,
            y_min=self._y_min,
            y_extent=self._y_extent,
        )

    def _find_available_tile_zoom_level(
        self,
        k,
        x,
        y,
        zoom_in_level,
        max_zoom_level,
    ):
        """
        Find an available tile zoom index for a label using the precomputed
        tile tree.

        Parameters
        ----------
        k : int
            Maximum labels per tile
        x, y : float
            Coordinates
        zoom_in_level : int
            Initial zoom index
        max_zoom_level : int
            Maximum allowed zoom level

        Returns
        -------
        int
            Available zoom index or -1 if none available
        """
        zoom_level = zoom_in_level

        assert self._tiles is not None

        while zoom_level <= max_zoom_level:
            tile = self._tiles.get(self._get_tile_id(x, y, zoom_level))

            # If the tile doesn't exist in our tree, we should return -1 because
            # higher zoom levels won't exist either based on how we constructed
            # the tree
            if tile is None:
                break

            # If the tile has space, use this zoom level
            if len(tile) < k:
                return zoom_level

            # Otherwise try the next zoom level
            zoom_level += 1

        # If we couldn't find a suitable tile at or above the requested zoom
        # level, try parent tiles at lower zoom levels
        zoom_level = zoom_in_level - 1
        while zoom_level >= 0:
            tile = self._tiles.get(self._get_tile_id(x, y, zoom_level))

            if tile is not None and len(tile) < k:
                return zoom_level

            # Otherwise try the next zoom level
            zoom_level -= 1

        # If we've exhausted our search and still couldn't find a tile with
        # space. Hence we will not display this label
        return -1

    def _tile_labels(
        self,
        labels: pd.DataFrame,
        k: int,
        min_zoom_extent: int = 1,
        max_zoom_level: int = MAX_ZOOM_LEVEL,
        progress_bar: Optional[tqdm] = None,
    ):
        """
        Adjust label zoom levels based on quad tiling approach to control density.
        Uses a precomputed tile tree to limit tile creation to necessary regions.

        Parameters
        ----------
        labels : pandas.DataFrame
            Label data with coordinates and dimensions
        k : int
            Maximum number of labels per tile
        min_zoom_extent : float, default=1
            Minimum zoom range a label must be visible for
        max_zoom_level : int, default=MAX_ZOOM_LEVEL
            Maximum zoom index

        Returns
        -------
        pandas.DataFrame
            Updated labels with adjusted zoom levels
        """

        if progress_bar:
            progress_bar.set_description('Tiling label data')

        # Build spatial index of all labels for efficient region queries
        xs = np.asarray(labels['x'].values)
        ys = np.asarray(labels['y'].values)

        builder = kdtree.KDTreeBuilder(num_items=len(labels))
        builder.add(xs, ys)
        label_index = builder.finish()

        import time

        t0 = time.perf_counter()

        # Precompute the tile tree
        self._tiles = {}

        # Precompute at initialization or at the beginning of _tile_labels
        zoom_scales = [2**z for z in range(max_zoom_level + 1)]
        tile_widths = [self._x_extent / scale for scale in zoom_scales]
        tile_heights = [self._y_extent / scale for scale in zoom_scales]

        # Start with base tiles at lowest zoom level
        base_zoom = 0
        base_tiles = [(0, 0, base_zoom)]

        # Process the tile queue
        tile_queue = base_tiles.copy()

        while tile_queue:
            x, y, z = tile_queue.pop(0)
            tile_id = f'{x},{y},{z}'

            # Calculate tile bounds in data coordinates
            x_min = self._x_min + x * tile_widths[z]
            x_max = self._x_min + (x + 1) * tile_widths[z]
            y_min = self._y_min + y * tile_heights[z]
            y_max = self._y_min + (y + 1) * tile_heights[z]

            # Determine how many points are contained within this tile
            num_labels_in_region = len(
                kdtree.range(label_index, x_min, y_min, x_max, y_max)
            )

            # Add this tile to our tree
            self._tiles[tile_id] = []

            # If more points than k and not at max zoom level, subdivide
            if num_labels_in_region > k and z < max_zoom_level:
                # Add child tiles to the queue
                _x = x * 2
                _y = y * 2
                _z = z + 1
                tile_queue.append((_x, _y, _z))
                tile_queue.append((_x + 1, _y, _z))
                tile_queue.append((_x, _y + 1, _z))
                tile_queue.append((_x + 1, _y + 1, _z))

        # Now that we have a precomputed tile tree, assign labels
        sorted_indices = labels.index.tolist()

        zoom_ins = np.asarray(labels['zoom_in'].values)
        zoom_outs = np.asarray(labels['zoom_out'].values)

        t1 = time.perf_counter()
        logger.info(f'Creating the tile tree took {t1 - t0} seconds')

        # Process labels in order of zoom-in level
        for i in sorted_indices:
            x, y = xs[i], ys[i]
            zoom_in = zoom_ins[i]

            # Find all tile zoom levels this label belongs to
            zoom_in_level = zoom_scale_to_zoom_level(zoom_in, max_zoom_level)

            # Find the available zoom level for this label
            available_zoom_in_level = self._find_available_tile_zoom_level(
                k, x, y, zoom_in_level, max_zoom_level
            )

            if available_zoom_in_level == -1:
                # Special case where no available tile could be found. Hide the
                # label altogether by setting its zoom in level to infinity
                zoom_ins[i] = math.inf
                zoom_outs[i] = 0
                continue

            if available_zoom_in_level > zoom_in_level:
                # Need to adjust the label's zoom_in
                zoom_ins[i] = zoom_level_to_zoom_scale(
                    available_zoom_in_level,
                    max_zoom_level,
                )
                zoom_outs[i] = max(
                    zoom_outs[i],
                    zoom_level_to_zoom_scale(
                        available_zoom_in_level + min_zoom_extent, max_zoom_level
                    ),
                )

            zoom_out = zoom_outs[i]
            zoom_out_level = zoom_scale_to_zoom_level(zoom_out, max_zoom_level)

            # Add label to all tiles it belongs to, but only if the tile exists
            # in our tree
            for zoom_level in range(
                available_zoom_in_level, min(zoom_out_level, max_zoom_level) + 1
            ):
                tile_id = self._get_tile_id(x, y, zoom_level)

                # Skip if this tile was not in our precomputed tree
                if tile_id not in self._tiles:
                    break

                self._tiles[tile_id].append(i)

            if progress_bar:
                progress_bar.update()

        labels['x'] = xs
        labels['y'] = ys
        labels['zoom_in'] = zoom_ins
        labels['zoom_out'] = zoom_outs

        t2 = time.perf_counter()
        logger.info(f'Assigning labels to tiles took {t2 - t1} seconds')

        return labels

    def _find_data_tile(self, tile_id: str):
        """
        Find the data tile for a given tile ID.

        Parameters
        ----------
        tile_id : str
            Tile ID

        Returns
        -------
        list of int
            Label indices for the data tile
        """
        assert self._tiles is not None

        if tile_id in self._tiles:
            return self._tiles[tile_id]

        x, y, z = map(lambda x: int(x), tile_id.split(','))

        max_xy = zoom_level_to_zoom_scale(z)

        if x < 0 or x > max_xy:
            return None

        if y < 0 or y > max_xy:
            return None

        while tile_id not in self._tiles and z >= 0:
            z -= 1
            x = math.floor(x / 2)
            y = math.floor(y / 2)
            tile_id = f'{x},{y},{z}'

        if tile_id in self._tiles:
            return self._tiles[tile_id]

        return None

    def _find_data_tiles(self, tile_ids: List[str]):
        """
        Find the data tiles for a list of tile IDs.

        Parameters
        ----------
        tile_ids : list of str
            Tile IDs

        Returns
        -------
        generator of list of int
            Label indices for the data tiles
        """
        for tile_id in tile_ids:
            data_tile = self._find_data_tile(tile_id)
            if data_tile is not None:
                yield data_tile

    def _get_label_idxs_from_tiles(self, tile_ids: List[str]) -> List[int]:
        return deduplicate(flatten(self._find_data_tiles(tile_ids)))

    def _get_total_num_operations(self):
        _, num_labels = self._get_filtered_labels()

        # Number of operations by method:
        # _create_labels: n
        # _compute_zoom_levels: n
        # _build_spatial_index: n
        # _resolve_collisions: n
        # _tile_labels: n

        return num_labels * 5

    def _compute_zoom_extent_fade_in_out(
        self, labels: pd.DataFrame, fade_extent_percentage: float = 0.1
    ):
        """
        Compute fade in/out extent for smooth label transitions.

        Parameters
        ----------
        labels : pd.DataFrame
            DataFrame with label information
        fade_extent_percentage : float, optional
            Percentage of zoom range to use for fading, by default 0.1

        Returns
        -------
        pd.DataFrame
            Updated labels with fade extents
        """
        zoom_ins = cast(npt.NDArray[np.float64], labels['zoom_in'].values)
        zoom_outs = cast(npt.NDArray[np.float64], labels['zoom_out'].values)

        zoom_fade_extent = np.zeros_like(zoom_ins)
        zoom_fade_extent = (
            np.subtract(
                zoom_outs,
                zoom_ins,
                out=zoom_fade_extent,
                where=np.logical_not(np.isinf(zoom_ins) | np.isinf(zoom_outs)),
            )
            * fade_extent_percentage
        )
        labels['zoom_fade_extent'] = zoom_fade_extent

        return labels

    def _get_filtered_labels(self, categorical_df: Optional[pd.DataFrame] = None):
        """
        Helper method to get filtered labels accounting for both
        regular categorical columns and point label columns.

        Parameters
        ----------
        categorical_df : pd.DataFrame, optional
            DataFrame containing the categorical columns

        Returns
        -------
        dict
            Dictionary mapping column names to dictionaries of label values and indices
        int
            Total number of labels
        """
        if categorical_df is None:
            categorical_df = cast(pd.DataFrame, self._data[self._by].copy())
            for column in self._by:
                if column in self._point_label_columns:
                    categorical_df[column] = categorical_df[column].astype('str')
                elif not is_categorical_data(categorical_df[column]):
                    categorical_df[column] = categorical_df[column].astype('category')

        filtered_grouped_data = {}
        label_count = 0

        for column in self._by:
            if column in self._exclude:
                # Skip this entire column if it's excluded
                continue

            filtered_groups = []

            if column in self._point_label_columns:
                # For point label columns, each row gets its own group
                for idx, value in enumerate(categorical_df[column]):
                    value_str = str(value)
                    label_type_value = f'{column}:{value_str}'

                    if label_type_value not in self._exclude:
                        # For point labels, each group has exactly one item
                        filtered_groups.append((value_str, [idx]))
                        label_count += 1
            else:
                # Regular grouping for normal categorical columns
                groups = categorical_df.groupby(column, observed=True).groups
                for label_value, indices in groups.items():
                    label_type_value = f'{column}:{label_value}'
                    if label_type_value not in self._exclude:
                        filtered_groups.append((label_value, indices))
                        label_count += 1

            filtered_grouped_data[column] = filtered_groups

        return filtered_grouped_data, label_count

    def _create_labels(self, progress_bar: Optional[tqdm] = None):
        """
        Preprocess data to extract label information.

        Returns
        -------
        pandas.DataFrame
            DataFrame with preprocessed label information.
        """

        if progress_bar:
            progress_bar.set_description('Preprocessing labels')

        # Convert only the needed categorical columns for faster filtering
        categorical_df = cast(pd.DataFrame, self._data[self._by].copy())
        for column in self._by:
            if column in self._point_label_columns:
                categorical_df[column] = categorical_df[column].astype('str')
            elif not is_categorical_data(categorical_df[column]):
                categorical_df[column] = categorical_df[column].astype('category')

        # Pre-extract coordinates as numpy arrays for faster access
        x_coords = cast(npt.NDArray[np.float64], self._data[self._x].values)
        y_coords = cast(npt.NDArray[np.float64], self._data[self._y].values)

        # Extract importance values if needed
        importance_values = cast(
            Optional[npt.NDArray[np.float64]],
            self._data[self._importance].values if self._importance else None,
        )

        # Get filtered grouped data
        filtered_grouped_data, label_count = self._get_filtered_labels(categorical_df)

        # Pre-allocate NumPy arrays for numerical data
        importances = np.zeros(label_count, dtype=np.float64)
        hashes = np.zeros(label_count, dtype=np.uint64)
        x_values = np.zeros(label_count, dtype=np.float64)
        y_values = np.zeros(label_count, dtype=np.float64)
        bbox_widths = np.zeros(label_count, dtype=np.float64)
        bbox_heights = np.zeros(label_count, dtype=np.float64)
        font_sizes = np.zeros(label_count, dtype=np.uint8)

        # Initialize lists for string data
        labels = [''] * label_count
        label_types = [''] * label_count
        font_faces = [''] * label_count
        font_styles = [''] * label_count
        font_weights = [''] * label_count
        font_colors = [''] * label_count

        # Process each hierarchy level
        i = 0
        for column in filtered_grouped_data:
            # Iterate through filtered groups instead of original groups
            for label, indices in filtered_grouped_data[column]:
                # Skip if no data points
                if len(indices) == 0:
                    continue

                label = str(label)

                indices = cast(List[int], indices)

                # Get the coordinates directly from pre-extracted arrays
                label_x_coords = x_coords[indices]
                label_y_coords = y_coords[indices]

                # Calculate importance
                if importance_values is None:
                    # Use number of points as importance
                    importance = len(indices)
                else:
                    # Use user-provided importance values
                    importance = aggregate(
                        importance_values[indices], self._importance_aggregation
                    )

                # Calculate hash for deterministic tiebreaking
                label_hash = self._compute_label_hash(label, indices)

                # Filter to percentile range if enough points
                if len(indices) > 3:
                    # Instead of separate filtering steps
                    x_lower, x_upper = np.percentile(
                        label_x_coords, self._bbox_percentile_range
                    )
                    y_lower, y_upper = np.percentile(
                        label_y_coords, self._bbox_percentile_range
                    )

                    # Filter coordinates with numpy operations
                    filter_mask = (
                        (label_x_coords >= x_lower)
                        & (label_x_coords <= x_upper)
                        & (label_y_coords >= y_lower)
                        & (label_y_coords <= y_upper)
                    )

                    filtered_x = label_x_coords[filter_mask]
                    filtered_y = label_y_coords[filter_mask]

                    # Create points array directly
                    if len(filtered_x) > 0:
                        points = np.column_stack((filtered_x, filtered_y))
                    else:
                        points = np.column_stack((label_x_coords, label_y_coords))
                else:
                    # Use all points
                    points = np.column_stack((label_x_coords, label_y_coords))

                # Compute hull, bounding box, and center of mass
                bbox, position = self._compute_bbox(points)

                # Fill arrays at position i
                label_types[i] = column
                importances[i] = importance
                hashes[i] = label_hash
                x_values[i] = position[0]
                y_values[i] = position[1]
                bbox_widths[i] = bbox[2] - bbox[0]
                bbox_heights[i] = bbox[3] - bbox[1]

                label_type_value = f'{column}:{remove_line_breaks(label)}'
                font_colors[i] = self._color.get(label_type_value, self._color[column])
                font_faces[i] = self._font.get(
                    label_type_value, self._font[column]
                ).face
                font_styles[i] = self._font.get(
                    label_type_value, self._font[column]
                ).style
                font_weights[i] = self._font.get(
                    label_type_value, self._font[column]
                ).weight
                font_sizes[i] = self._size.get(label_type_value, self._size[column])

                if self._target_aspect_ratio is not None:
                    label = optimize_line_breaks(
                        text=label,
                        text_measurer=self._text.get(
                            label_type_value,
                            self._text[column],
                        ),
                        font_size=font_sizes[i],
                        target_aspect_ratio=self._target_aspect_ratio,
                        max_lines=self._max_lines,
                    )

                labels[i] = label

                i += 1

                if progress_bar:
                    progress_bar.update()

        # Create DataFrame from collected data
        labels = pd.DataFrame(
            {
                'label': labels,
                'label_type': pd.Categorical(
                    values=label_types,
                    categories=self._by,
                    ordered=True,
                ),
                'importance': importances,
                'hash': hashes,
                'x': x_values,
                'y': y_values,
                'bbox_width': bbox_widths,
                'bbox_height': bbox_heights,
                'font_color': pd.Categorical(font_colors),
                'font_face': pd.Categorical(font_faces),
                'font_style': pd.Categorical(font_styles),
                'font_weight': pd.Categorical(font_weights),
                'font_size': pd.Categorical(font_sizes),
            }
        )

        # Sort by hierarchy and importance
        if not labels.empty:
            if self._hierarchical:
                sort_columns = ['label_type', 'importance', 'hash']
                sort_order = [True, False, True]
            else:
                sort_columns = ['importance', 'hash']
                sort_order = [False, True]

            labels.sort_values(
                by=sort_columns,
                ascending=sort_order,
                inplace=True,
            )

        labels.reset_index(drop=True, inplace=True)

        return labels
