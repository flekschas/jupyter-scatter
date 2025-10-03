from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import numpy.typing as npt
from geoindex_rs import rtree

from .constants import (
    NUM_LABELS_SOLVE_ZOOM_LEVELS_APPROXIMATELY,
    NUM_LABELS_SOLVE_ZOOM_LEVELS_PRECISELY,
)
from ..dependencies import MissingCallable
from .k import compute_k
from .utils import noop
from .zoom_solver import solve_zoom, solve_zoom_approximately, solve_zoom_precisely

if TYPE_CHECKING:
    from tqdm.auto import tqdm
else:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = MissingCallable.class_('tqdm', 'tqdm.auto', 'label-extras')


def resolve_static_zoom_in_collisions(
    spatial_index: Any,
    center_xs: npt.NDArray[np.float64],
    center_ys: npt.NDArray[np.float64],
    half_widths: npt.NDArray[np.float64],
    half_heights: npt.NDArray[np.float64],
    min_xs: npt.NDArray[np.float64],
    min_ys: npt.NDArray[np.float64],
    max_xs: npt.NDArray[np.float64],
    max_ys: npt.NDArray[np.float64],
    zoom_ins: npt.NDArray[np.float64],
    zoom_outs: npt.NDArray[np.float64],
    idxs: npt.NDArray[np.uint64],
    progress_bar: Optional[tqdm] = None,
):
    """
    Resolve collisions between static labels when zooming in

    Parameters
    ----------
    labels : pandas.DataFrame
        Label data with coordinates and dimensions
    spatial_idx : Spatial index for collision detection
    progress_bar : tqdm.tqdm, optional
        Progress bar for tracking computation

    Returns
    -------
    pandas.DataFrame
        Updated labels with collision-free zoom levels
    """

    update_progress_bar = progress_bar.update if progress_bar else noop

    # Since we can skip the first label, we're going to update the progress bar
    # immediately
    update_progress_bar()

    # Process labels in order of priority
    for i in range(1, len(idxs)):
        idx = idxs[i]

        # Query for intersections
        intersecting_idx = rtree.search(
            spatial_index, min_xs[idx], min_ys[idx], max_xs[idx], max_ys[idx]
        ).to_numpy()

        # Only consider higher priority labels (those with lower indices)
        collisions = intersecting_idx[intersecting_idx < idx]

        if len(collisions) == 0:
            update_progress_bar()
            continue

        # Extract zoom out values for all colliding labels at once
        colliding_zoom_outs = zoom_outs[collisions]

        # Visibility range overlap check: we only need to consider
        # collisions where the lower priority label `i` is shown before
        # higher priority colliding labels are fade (zoomed) out
        collisions = collisions[zoom_ins[idx] < colliding_zoom_outs]

        if len(collisions) == 0:
            update_progress_bar()
            continue

        # Extract current label properties
        center_x = center_xs[idx]
        center_y = center_ys[idx]
        half_width = half_widths[idx]
        half_height = half_heights[idx]

        # Extract data for all colliding labels at once
        colliding_centers_x = center_xs[collisions]
        colliding_centers_y = center_ys[collisions]
        colliding_half_widths = half_widths[collisions]
        colliding_half_heights = half_heights[collisions]

        # Refilter zoom out values colliding data
        colliding_zoom_outs = zoom_outs[collisions]

        # Compute distance and dimension
        widths = half_width + colliding_half_widths
        heights = half_height + colliding_half_heights
        d_xs = np.abs(center_x - colliding_centers_x)
        d_ys = np.abs(center_y - colliding_centers_y)

        # Calculate resolution zoom levels
        x_resolutions = np.divide(
            widths, d_xs, out=np.full_like(d_xs, np.inf), where=d_xs > 0
        )
        y_resolutions = np.divide(
            heights, d_ys, out=np.full_like(d_ys, np.inf), where=d_ys > 0
        )

        # The final zoom scale at which the collision is resolved
        resolution_zooms = np.minimum(x_resolutions, y_resolutions)

        # Since some colliding labels might be faded/zoomed out already
        # before the collision would be resolved, we can take the minimum
        # between the two
        resolution_zooms = np.minimum(resolution_zooms, colliding_zoom_outs)

        # Finally, we're going to take the maximum zoom scale to ensure all
        # collisions are resolved
        zoom_ins[idx] = np.max(resolution_zooms)

        update_progress_bar()

    return zoom_ins, zoom_outs


def resolve_asinh_zoom_in_collisions(
    spatial_index: Any,
    center_xs: npt.NDArray[np.float64],
    center_ys: npt.NDArray[np.float64],
    half_widths: npt.NDArray[np.float64],
    half_heights: npt.NDArray[np.float64],
    min_xs: npt.NDArray[np.float64],
    min_ys: npt.NDArray[np.float64],
    max_xs: npt.NDArray[np.float64],
    max_ys: npt.NDArray[np.float64],
    zoom_ins: npt.NDArray[np.float64],
    zoom_outs: npt.NDArray[np.float64],
    idxs: npt.NDArray[np.uint64],
    progress_bar: Optional[tqdm] = None,
):
    """
    Resolve collisions between asinh-scaled labels when zooming in

    Parameters
    ----------
    labels : pandas.DataFrame
        Label data with coordinates and dimensions
    spatial_idx : Spatial index for collision detection
    progress_bar : tqdm.tqdm, optional
        Progress bar for tracking computation

    Returns
    -------
    pandas.DataFrame
        Updated labels with collision-free zoom levels
    """

    n = len(center_xs)

    # Choose the appropriate solver based on number of labels
    if n < NUM_LABELS_SOLVE_ZOOM_LEVELS_PRECISELY:
        solver = solve_zoom_precisely
    elif n > NUM_LABELS_SOLVE_ZOOM_LEVELS_APPROXIMATELY:
        solver = solve_zoom_approximately
    else:
        solver = solve_zoom

    update_progress_bar = progress_bar.update if progress_bar else noop

    # Since we can skip the first label, we're going to update the progress bar
    # immediately
    update_progress_bar()

    # Process labels in order of priority
    for i in np.arange(1, n):
        # Query for intersections
        intersecting_idx = rtree.search(
            spatial_index, min_xs[i], min_ys[i], max_xs[i], max_ys[i]
        ).to_numpy()

        # Only consider higher priority labels (those with lower indices)
        collisions = intersecting_idx[intersecting_idx < idxs[i]]

        # Extract current label properties
        center_x = center_xs[i]
        center_y = center_ys[i]
        half_width = half_widths[i]
        half_height = half_heights[i]

        # Non-vectorized approach for other scale functions
        for colliding_idx in collisions:
            colliding_center_x = center_xs[colliding_idx]
            colliding_center_y = center_ys[colliding_idx]
            colliding_half_width = half_widths[colliding_idx]
            colliding_half_height = half_heights[colliding_idx]

            # If there's no overlap in zoom visibility, skip this collision
            if zoom_ins[i] >= zoom_outs[colliding_idx]:
                continue

            # Handle edge case of identical positions
            if (
                abs(center_x - colliding_center_x) < 1e-10
                and abs(center_y - colliding_center_y) < 1e-10
            ):
                zoom_ins[i] = np.float64(math.inf)
                zoom_outs[i] = np.float64(math.inf)
                break  # No need to check other collisions

            w = half_width + colliding_half_width
            h = half_height + colliding_half_height
            d_x = np.abs(center_x - colliding_center_x)
            d_y = np.abs(center_y - colliding_center_y)

            x_constant_resolution = w / d_x if d_x else math.inf
            y_constant_resolution = h / d_y if d_y else math.inf

            # Calculate resolution zoom level using the solver
            if x_constant_resolution < y_constant_resolution:
                k = compute_k(
                    center_x,
                    half_width,
                    colliding_center_x,
                    colliding_half_width,
                )
            else:
                k = compute_k(
                    center_y,
                    half_height,
                    colliding_center_y,
                    colliding_half_height,
                )

            try:
                resolution_zoom = solver(k)
            except Exception:
                # Fallback if solver fails
                resolution_zoom = min(x_constant_resolution, y_constant_resolution)

            # Update zoom level
            if resolution_zoom > zoom_outs[colliding_idx]:
                if zoom_outs[i] > zoom_outs[colliding_idx]:
                    # Since the zoom out level is greater than the colliding
                    # zoom out level, we can show this label when the
                    # colliding (higher priority) label gets hidden
                    zoom_ins[i] = zoom_outs[colliding_idx]
                else:
                    # Since the zoom out level of the label is lower than
                    # the zoom out level of the colliding (higher priority)
                    # label, we have to hide this label
                    zoom_ins[i] = math.inf
            else:
                zoom_ins[i] = max(zoom_ins[i], resolution_zoom)

        update_progress_bar()

    return zoom_ins, zoom_outs
