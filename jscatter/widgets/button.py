import anywidget

from traitlets import Bool, Int, Unicode


class Button(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
      const button = document.createElement('button');

      button.classList.add(
        'jupyter-widgets',
        'jupyter-button',
        'widget-button',
        'jupyter-scatter-button',
      );

      const update = () => {
        const description = model.get('description');
        const icon = model.get('icon');
        const tooltip = model.get('tooltip');
        const width = model.get('width');

        button.textContent = '';

        if (icon.startsWith('<svg')) {
          const parser = new DOMParser();
          const doc = parser.parseFromString(icon, "image/svg+xml");
          button.appendChild(doc.firstChild);
        } else {
          const i = document.createElement('i');
          i.classList.add('fa', `fa-${icon}`);

          if (!description) {
            i.classList.add('center');
          }

          button.appendChild(i);
        }

        if (description) {
          button.appendChild(document.createTextNode(description));
        }

        if (tooltip) {
          button.title = tooltip;
        }

        if (width) {
          button.style.width = `${width}px`;
        }
      }

      const createEventHandler = (eventType) => (event) => {
        model.send({
          type: eventType,
          alt_key: event.altKey,
          shift_key: event.shiftKey,
          meta_key: event.metaKey,
        });
      }

      const clickHandler = createEventHandler('click');
      const dblclickHandler = createEventHandler('dblclick');

      button.addEventListener('click', clickHandler);
      button.addEventListener('dblclick', dblclickHandler);

      model.on('change:description', update);
      model.on('change:icon', update);
      model.on('change:width', update);
      model.on('change:tooltip', update);

      update();

      const updateVisibility = () => {
        const visible = model.get('visible');
        if (visible) {
          el.style.display = 'block';
        } else {
          el.style.display = 'none';
        }
      }

      model.on('change:visible', updateVisibility);

      updateVisibility();

      el.appendChild(button);

      return () => {
        button.removeEventListener('click', clickHandler);
        button.removeEventListener('dblclick', dblclickHandler);
      };
    }
    export default { render }
    """

    _css = """
    .jupyter-scatter-button {
      position: relative;
    }
    .jupyter-scatter-button > svg {
      display: block;
      width: 100%;
      height: 100%;
    }
    """

    description = Unicode().tag(sync=True)
    icon = Unicode().tag(sync=True)
    width = Int(allow_none=True).tag(sync=True)
    visible = Bool(True).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._click_handler = None
        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, content: dict, buffers):
        if content['type'] == 'click' and self._click_handler is not None:
            self._click_handler(content)
        if content['type'] == 'dblclick' and self._dblclick_handler is not None:
            self._dblclick_handler(content)

    def on_click(self, callback):
        self._click_handler = callback

    def on_dblclick(self, callback):
        self._dblclick_handler = callback
