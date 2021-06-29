from ipywidgets.widgets import GridBox, Layout
from itertools import zip_longest

def compose(
    scatters,
    select_mappers = None,
    hover_mappers = None,
    row_height = 320,
):
    n_rows = 1

    if all(isinstance(el, list) for el in scatters):
        # Nested lists
        n_rows = len(scatters)
        n_cols = len(scatters[0])
    else:
        n_cols = len(scatters)

    try:
        scatters = [item for sublist in scatters for item in sublist]
    except TypeError:
        # Probably not a nested list
        pass

    try:
        select_mappers = [item for sublist in select_mappers for item in sublist]
    except TypeError:
        # Probably not a nested list
        pass

    try:
        hover_mappers = [item for sublist in hover_mappers for item in sublist]
    except TypeError:
        # Probably not a nested list
        pass

    complete_scatters = list(zip_longest(
        scatters,
        select_mappers or [],
        hover_mappers or []
    ))

    assert len(scatters) == n_rows * n_cols, 'Should equal'

    # We need to store the specific handlers created by called
    # `base_select_handler(index)` and `base_hover_handler(index)` to
    # dynamically un- and re-observe
    scatter_select_handlers = []
    scatter_hover_handlers = []

    def base_select_handler(index):
        def select_handler(change):
            if change['new'] is None:
                for i, complete_scatter in enumerate(complete_scatters):
                    if i != index:
                        sc, _, _ = complete_scatter
                        # Unsubscribe to avoid cyclic updates
                        sc.widget.unobserve(scatter_select_handlers[i], names='selection')
                        sc.widget.selection = []
                        # Re-subscribe to listen to changes coming from the JS kernel
                        sc.widget.observe(scatter_select_handlers[i], names='selection')

            else:
                for i, complete_scatter in enumerate(complete_scatters):
                    if i != index:
                        sc, select_mapper, _ = complete_scatter
                        # Unsubscribe to avoid cyclic updates
                        sc.widget.unobserve(scatter_select_handlers[i], names='selection')

                        if select_mapper is None:
                            sc.widget.selection = [int(x) for x in change['new']]
                        else:
                            sc.widget.selection = [int(x) for x in select_mapper(index, change['new'])]

                        # Re-subscribe to listen to changes coming from the JS kernel
                        sc.widget.observe(scatter_select_handlers[i], names='selection')

        return select_handler

    def base_hover_handler(index):
        def hover_handler(change):
            if change['new'] is None:
                for i, complete_scatter in enumerate(complete_scatters):
                    if i != index:
                        sc, _, _ = complete_scatter
                        # Unsubscribe to avoid cyclic updates
                        sc.widget.unobserve(scatter_hover_handlers[i], names='hovering')
                        sc.widget.hovering = -1
                        # Re-subscribe to listen to changes coming from the JS kernel
                        sc.widget.observe(scatter_hover_handlers[i], names='hovering')

            else:
                for i, complete_scatter in enumerate(complete_scatters):
                    if i != index:
                        sc, _, hover_mapper = complete_scatter
                        # Unsubscribe to avoid cyclic updates
                        sc.widget.unobserve(scatter_hover_handlers[i], names='hovering')

                        if hover_mapper is None:
                            sc.widget.hovering = int(change['new'])
                        else:
                            sc.widget.hovering = int(hover_mapper(index, change['new']))

                        # Re-subscribe to listen to changes coming from the JS kernel
                        sc.widget.observe(scatter_hover_handlers[i], names='hovering')

        return hover_handler

    for index, scatter in enumerate(scatters):
        scatter.height(row_height)
        scatter.widget.unobserve_all() # Clear previous observers

        select_handler = base_select_handler(index)
        hover_handler = base_hover_handler(index)

        scatter_select_handlers.append(select_handler)
        scatter_hover_handlers.append(hover_handler)

        scatter.widget.observe(select_handler, names='selection')
        scatter.widget.observe(hover_handler, names='hovering')

    return GridBox(
        children=[scatter.widget.show() for scatter in scatters],
        layout=Layout(
            grid_template_columns=' '.join(['auto' for x in range(n_cols)]),
            grid_template_rows=' '.join([f'{row_height}px' for x in range(n_rows)]),
            grid_gap='2px'
        )
    )
