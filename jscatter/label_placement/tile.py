import math

from .constants import MAX_ZOOM_LEVEL


def zoom_scale_to_zoom_level(zoom_scale: float, max_zoom_level=MAX_ZOOM_LEVEL):
    """
    Translate zoom scale to tile zoom index according to 2^x.

    Args:
        zoom: Zoom scale

    Returns:
        Tile zoom index
    """
    if zoom_scale <= 1:
        return 0

    if math.isinf(zoom_scale):
        return max_zoom_level

    return math.floor(math.log2(zoom_scale))


def zoom_level_to_zoom_scale(zoom_level: int, max_zoom_level=MAX_ZOOM_LEVEL):
    """
    Translate zoom level to zoom scale according to `2 ** x` where `x` is the
    zoom level.

    Args:
        zoom_level: Zoom level

    Returns:
        Zoom scale
    """
    return 2 ** max(0, min(zoom_level, max_zoom_level))


def get_tile_id(x, y, x_min, x_extent, y_min, y_extent, zoom_level):
    """
    Generate a tile key based on coordinates and zoom level.

    Args:
        x, y: Coordinates
        x_min, x_extent, y_min, y_extent: Domain information
        zoom_level: Zoom index

    Returns:
        Tile key string
    """
    zoom_scale = zoom_level_to_zoom_scale(zoom_level)

    if x_extent == 0:
        tile_x = 0
    else:
        tile_x = max(
            0, min(zoom_scale - 1, math.floor((x - x_min) / (x_extent / zoom_scale)))
        )

    if y_extent == 0:
        tile_y = 0
    else:
        tile_y = max(
            0, min(zoom_scale - 1, math.floor((y - y_min) / (y_extent / zoom_scale)))
        )

    return f'{tile_x},{tile_y},{zoom_level}'
