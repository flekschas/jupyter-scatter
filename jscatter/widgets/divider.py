import anywidget


class Divider(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
      const div = document.createElement('div');
      div.classList.add('jupyter-widgets', 'jupyter-scatter-divider');
      el.appendChild(div);
    }
    export default { render }
    """

    _css = """
    .jupyter-scatter-divider {
      margin: 10px 0;
      width: 100%;
      height: 0;
      border: 1px solid var(--jp-layout-color2);
    }
    """
