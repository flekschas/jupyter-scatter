from uuid import uuid4
from ipywidgets.widgets import GridBox, Layout
from itertools import zip_longest
from typing import List, Optional, Union

from .jscatter import Scatter

def compose(
    scatters: List[Scatter],
    sync_view: bool = False,
    sync_selection: bool = False,
    sync_hover: bool = False,
    match_by: Union[str, List[str]] = 'index',
    rows: Optional[int] = 1,
    row_height: int = 320,
    cols: Optional[int] = None,
):
    if rows is not None and cols is not None:
        assert len(scatters) == rows * cols
    elif cols is not None:
        assert len(scatters) % cols == 0
        rows = len(scatters) // cols
    elif rows is not None:
        assert len(scatters) % rows == 0
        cols = len(scatters) // rows
    else:
        cols = len(scatters)
        rows = 1

    if isinstance(match_by, list):
        assert len(scatters) == len(match_by), 'The number of scatters and match_bys need to be the same'
    elif match_by != 'index':
        match_by = [match_by] * len(scatters)

    if isinstance(match_by, list):
        assert all([match_by[i] in sc._data for i, sc in enumerate(scatters)])

    # We need to store the specific handlers created by called
    # `create_select_handler(index)` and `create_hover_handler(index)` to
    # dynamically un- and re-observe
    scatter_select_handlers = []
    scatter_hover_handlers = []

    def create_idx_matcher(index: int, multiple: bool = False):
        def match_single(scatter, change):
            if change['new'] is None:
                return -1

            idx = int(change['new'])

            if match_by == 'index':
                return idx

            try:
                return scatter._data.iloc[[idx]][match_by[index]].unique()[0]
            except IndexError:
                return -1

        def match_multiple(scatter, change):
            if change['new'] is None:
                return []

            idxs = [int(x) for x in change['new']]

            if match_by == 'index':
                return idxs

            return scatter._data.iloc[idxs][match_by[index]].unique()

        return match_multiple if multiple else match_single

    def create_match_mapper(index: int, multiple: bool = False):
        def map_single(scatter, matched_id):
            if match_by == 'index':
                return matched_id

            try:
                return scatter._data.query(f'{match_by[index]} == {matched_id}').index.tolist()[0]
            except IndexError:
                return -1

        def map_multiple(scatter, matched_ids):
            if match_by == 'index':
                return matched_ids

            return scatter._data.query(f'{match_by[index]} in @matched_ids').index.tolist()

        return map_multiple if multiple else map_single

    def create_select_handler(index: int):
        idx_matcher = create_idx_matcher(index, multiple=True)
        match_mapper = create_match_mapper(index, multiple=True)

        def select_handler(change):
            matched_ids = idx_matcher(scatters[index], change)

            for i, scatter in enumerate(scatters):
                if i == index:
                    # Hover event was triggered by this widget, hence we
                    # don't need to update it
                    continue

                # Unsubscribe to avoid cyclic updates
                scatter.widget.unobserve(scatter_select_handlers[i], names='selection')

                if change['new'] is None:
                    scatter.widget.selection = []
                elif match_by == 'index':
                    scatter.widget.selection = matched_ids
                else:
                    scatter.widget.selection = match_mapper(scatter, matched_ids)

                # Re-subscribe to listen to changes coming from the JS kernel
                scatter.widget.observe(scatter_select_handlers[i], names='selection')

        return select_handler

    def create_hover_handler(index: int):
        idx_matcher = create_idx_matcher(index)
        match_mapper = create_match_mapper(index)

        def hover_handler(change):
            matched_id = idx_matcher(scatters[index], change)

            for i, scatter in enumerate(scatters):
                if i == index:
                    continue

                # Unsubscribe to avoid cyclic updates
                scatter.widget.unobserve(scatter_hover_handlers[i], names='hovering')

                if change['new'] is None:
                    scatter.widget.hovering = -1
                elif match_by == 'index':
                    scatter.widget.hovering = matched_id
                else:
                    scatter.widget.hovering = match_mapper(scatter, matched_id)

                # Re-subscribe to listen to changes coming from the JS kernel
                scatter.widget.observe(scatter_hover_handlers[i], names='hovering')

        return hover_handler

    for index, scatter in enumerate(scatters):
        scatter.height(row_height)
        scatter.widget.unobserve_all() # Clear previous observers

        select_handler = create_select_handler(index)
        hover_handler = create_hover_handler(index)

        scatter_select_handlers.append(select_handler)
        scatter_hover_handlers.append(hover_handler)

        if sync_selection:
            scatter.widget.observe(select_handler, names='selection')
        if sync_hover:
            scatter.widget.observe(hover_handler, names='hovering')

    if sync_view:
        uuid = uuid4().hex
        for scatter in scatters:
            scatter.widget.view_sync = uuid

    return GridBox(
        children=[scatter.widget.show() for scatter in scatters],
        layout=Layout(
            grid_template_columns=' '.join(['1fr' for x in range(cols)]),
            grid_template_rows=' '.join([f'{row_height}px' for x in range(rows)]),
            grid_gap='2px'
        )
    )

def link(
    scatters: List[Scatter],
    match_by: Union[str, List[str]] = 'index',
    rows: Optional[int] = 1,
    row_height: int = 320,
    cols: Optional[int] = None
):
    return compose(
        scatters,
        match_by=match_by,
        rows=rows,
        row_height=row_height,
        cols=cols,
        sync_view=True,
        sync_selection=True,
        sync_hover=True,
    )
