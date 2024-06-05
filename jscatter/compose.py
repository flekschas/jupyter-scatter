from uuid import uuid4
from ipywidgets.widgets import GridBox, HTML, Layout, VBox
from itertools import zip_longest
from typing import List, Optional, Union, Tuple

from .jscatter import Scatter

TITLE_HEIGHT = 28;
AXES_LABEL_SIZE = 16;
AXES_PADDING_Y = 20;

def compose(
    scatters: Union[List[Scatter], List[Tuple[Scatter, str]]],
    sync_view: bool = False,
    sync_selection: bool = False,
    sync_hover: bool = False,
    match_by: Union[str, List[str]] = 'index',
    rows: Optional[int] = None,
    row_height: int = 320,
    cols: Optional[int] = None,
):
    """
    Compose multiple `Scatter` instances and optionally synchronize their view,
    selection, and hover state.

    Parameters
    ----------
    scatters : list of Scatter, optional
        A list of `Scatter` instances
    sync_view : boolean, optional
        If `True` the views of all `Scatter` instances are synchronized
    sync_selection : boolean, optional
        If `True` the selection of all `Scatter` instances are synchronized
    sync_hover : boolean, optional
        If `True` the hover state of all `Scatter` instances are synchronized
    match_by : str, optional
        A string referencing a categorical column in `data` that specifies the
        point correspondences. Defaults to the DataFrame's index.
    rows : int, optional
        The number of rows. Defaults to `1` when unspecified.
    row_height: int, optional
        The row height in pixels. Defaults to `320`.
    cols : int, optional
        The number of columns. Defaults to `len(scatters)` when unspecified

    Returns
    -------
    GridBox
        An `ipywidget.GridBox` widget

    See Also
    --------
    link : Compose and link multiple scatter plot instances

    Examples
    --------
    >>> compose([scatter_a, scatter_b], cols=1, sync_selection=True)
    <ipywidget.ipywidget.GridBox>
    """
    if rows is not None and cols is not None:
        assert len(scatters) <= rows * cols
    elif cols is not None:
        rows = max(1, len(scatters) // cols)
    elif rows is not None:
        cols = max(1, len(scatters) // rows)
    else:
        cols = len(scatters)
        rows = 1

    if isinstance(match_by, list):
        assert len(scatters) == len(match_by), 'The number of scatters and match_bys need to be the same'
    elif match_by != 'index':
        match_by = [match_by] * len(scatters)

    has_titles = any([isinstance(scatter, tuple) for scatter in scatters])

    def get_scatter(i: int) -> Scatter:
        if has_titles and isinstance(scatters[i], tuple):
            return scatters[i][0]
        return scatters[i]

    def get_title(i: int) -> str:
        if has_titles and isinstance(scatters[i], tuple):
            return scatters[i][1]
        return "&nbsp;"

    if isinstance(match_by, list):
        assert all([match_by[i] in get_scatter(i)._data for i, _ in enumerate(scatters)])

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
                return scatter._data.query(f'{match_by[index]} == @matched_id').index.tolist()[0]
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
            matched_ids = idx_matcher(get_scatter(index), change)

            for i, _ in enumerate(scatters):
                if i == index:
                    # Hover event was triggered by this widget, hence we
                    # don't need to update it
                    continue

                scatter = get_scatter(i)

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

        select_handler.__jscatter_compose_observer__ = True

        return select_handler

    def create_hover_handler(index: int):
        idx_matcher = create_idx_matcher(index)
        match_mapper = create_match_mapper(index)

        def hover_handler(change):
            matched_id = idx_matcher(get_scatter(index), change)

            for i, _ in enumerate(scatters):
                if i == index:
                    continue

                scatter = get_scatter(i)

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

        hover_handler.__jscatter_compose_observer__ = True

        return hover_handler

    has_axes = any([
        get_scatter(i)._axes != False
        for i, _
        in enumerate(scatters)
    ])
    y_padding = AXES_PADDING_Y if has_axes else 0

    if has_axes:
        has_labels = any([
            get_scatter(i)._axes_labels != False
            for i, _
            in enumerate(scatters)
        ])
        y_padding = y_padding + AXES_LABEL_SIZE if has_labels else y_padding

    y_padding = y_padding + TITLE_HEIGHT if has_titles else y_padding

    for i, _ in enumerate(scatters):
        scatter = get_scatter(i)
        scatter.height(row_height - y_padding)

        trait_notifiers = scatter.widget._trait_notifiers

        # Clear previous `selection` and `hovering` observers
        for name in ['selection', 'hovering']:
            if name in trait_notifiers and 'change' in trait_notifiers[name]:
                for observer in trait_notifiers[name]['change']:
                    if hasattr(observer, '__jscatter_compose_observer__'):
                        trait_notifiers[name]['change'].remove(observer)

        select_handler = create_select_handler(i)
        hover_handler = create_hover_handler(i)

        scatter_select_handlers.append(select_handler)
        scatter_hover_handlers.append(hover_handler)

        if sync_selection:
            scatter.widget.observe(select_handler, names='selection')
        if sync_hover:
            scatter.widget.observe(hover_handler, names='hovering')

    if sync_view:
        uuid = uuid4().hex
        for i, _ in enumerate(scatters):
            get_scatter(i).widget.view_sync = uuid

    def get_scatter_widget(i):
        scatter_widget = get_scatter(i).widget.show()
        if has_titles:
            title = HTML(
                value=f'<b style="display: flex; justify-content: center; margin: 0 0 0 38px;">{get_title(i)}</b>',
            )
            return VBox([title, scatter_widget])
        return scatter_widget

    return GridBox(
        children=[get_scatter_widget(i) for i, _ in enumerate(scatters)],
        layout=Layout(
            grid_template_columns=' '.join(['1fr' for x in range(cols)]),
            grid_template_rows=' '.join([f'{row_height}px' for x in range(rows)]),
            grid_gap='2px'
        )
    )

def link(
    scatters: Union[List[Scatter], List[Tuple[Scatter, str]]],
    match_by: Union[str, List[str]] = 'index',
    rows: Optional[int] = 1,
    row_height: int = 320,
    cols: Optional[int] = None
):
    """
    A short-hand function for `compose` that composes multiple `Scatter`
    instances and automatically synchronizes their view, selection, and hover
    state.

    Parameters
    ----------
    scatters : list of Scatter, optional
        A list of `Scatter` instances
    match_by : str, optional
        A string referencing a categorical column in `data` that specifies the
        point correspondences. Defaults to the DataFrame's index.
    rows : int, optional
        The number of rows. Defaults to `1` when unspecified.
    row_height: int, optional
        The row height in pixels. Defaults to `320`.
    cols : int, optional
        The number of columns. Defaults to `len(scatters)` when unspecified

    Returns
    -------
    GridBox
        An `ipywidget.GridBox` widget

    See Also
    --------
    compose : Compose multiple scatter plot instances

    Examples
    --------
    >>> link([Scatter(data=df_a, x='x', y='y'), Scatter(data=df_a, x='x', y='y')])
    <ipywidget.ipywidget.GridBox>
    """
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
