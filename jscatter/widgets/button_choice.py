import anywidget

from traitlets import Dict, Int, List, Unicode, Union


class ButtonChoice(anywidget.AnyWidget):
    _esm = """
    function removeAllChildren(node) {
      while (node.firstChild) {
        node.removeChild(node.firstChild);
      }
    }

    function render({ model, el }) {
      const id = Array.from({ length: 8 }, () =>
        'abcdefghijklmnopqrstuvwxyz'.charAt(Math.floor(Math.random() * 26))
      ).join('');

      const button = document.createElement('button');
      const dialog = document.createElement('dialog');
      const container = document.createElement('div');

      button.classList.add('jupyter-widgets', 'jupyter-button', 'widget-button', 'jupyter-scatter-choice-toggler');
      dialog.classList.add('jupyter-scatter-choice-dialog');
      container.classList.add('jupyter-widgets', 'jupyter-scatter-choice-container');

      let open = false;

      const updateIcon = (newValue) => {
        const icons = model.get('icon');
        const value = newValue || model.get('value');

        const icon = icons
          ? typeof icons === 'string'
            ? icons
            : icons[value]
          : undefined;

        if (icon) {
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
        }
      }

      const update = () => {
        const value = model.get('value');
        const options = model.get('options');
        const description = model.get('description');
        const tooltip = model.get('tooltip');
        const width = model.get('width');

        button.textContent = '';

        updateIcon();

        if (description) {
          button.appendChild(document.createTextNode(description));
        }

        button.title = tooltip || '';

        if (width) {
          button.style.width = `${width}px`;
        }

        if (container.firstChild) {
          container.querySelectorAll('input').forEach((input) => {
            input.removeEventListener('change', changeHandler);
          });
        }

        removeAllChildren(container);

        const optionValueLabelPairs = Array.isArray(options)
          ? options.map((option) => [option, option])
          : Object.entries(options)
        const template = optionValueLabelPairs.reduce((t, [value, label]) => {
          t += `<label class="jupyter-scatter-choice-label"><input id="${id}-${value}" type="radio" name="${id}" value="${value}" />${label}</label>`
          return t;
        }, '');

        container.insertAdjacentHTML('beforeend', template);
        container.querySelectorAll('input').forEach((input) => {
          input.checked = input.value === value;
          input.addEventListener('change', changeHandler);
        });
      }

      const dialogClickHandler = (event) => {
        if (event.target === dialog) {
          dialog.close();
        }
      }

      dialog.addEventListener('click', dialogClickHandler);

      const clickHandler = () => {
        const { top, left } = button.getBoundingClientRect();
        dialog.style.top = `${top}px`;
        dialog.style.left = `${left + (model.get('width') || 0)}px`;
        dialog.showModal();
      }

      const changeHandler = (event) => {
        updateIcon(event.target.value);
        model.set('value', event.target.value);
        model.save_changes();
        dialog.close();
      }

      button.addEventListener('click', clickHandler);

      model.on('change:description', update);
      model.on('change:icon', update);
      model.on('change:width', update);
      model.on('change:tooltip', update);
      model.on('change:value', update);
      model.on('change:options', update);

      update();

      el.appendChild(button);
      dialog.appendChild(container);
      el.appendChild(dialog);

      return () => {
        button.removeEventListener('click', clickHandler);
        dialog.removeEventListener('click', dialogClickHandler);
        container.querySelectorAll('input').forEach((input) => {
          input.removeEventListener('change', changeHandler);
        });
      };
    }
    export default { render }
    """

    _css = """
    .jupyter-scatter-choice-toggler {
      position: relative;
    }
    .jupyter-scatter-choice-toggler > svg {
      display: block;
      width: 100%;
      height: 100%;
    }
    .jupyter-scatter-choice-dialog {
      position: absolute;
      padding: 0;
      margin: 0;
      border: none;
    }
    .jupyter-scatter-choice-container {
      width: min-content;
      display: flex;
      align-items: center;
      gap: 0 1rem;
      padding: 0 0.5rem;
      white-space: nowrap;
      overflow: hidden;
      font-size: var(--jp-widgets-font-size);
      height: var(--jp-widgets-inline-height);
      line-height: var(--jp-widgets-inline-height);
      color: var(--jp-ui-font-color1);
      background-color: var(--jp-layout-color2);
      border-radius: 0 0.25rem 0.25rem 0;
      user-select: none;
      border: none;
      margin: 0;
    }
    .jupyter-scatter-choice-label {
      display: flex;
      align-items: center;
      gap: 0 0.25rem;
    }
    ::backdrop{
      background: var(--jp-layout-color0);
      opacity: 0.33;
    }
    """

    value = Unicode().tag(sync=True)
    options = Union([Dict(), List()], allow_none=True).tag(sync=True)
    description = Unicode(default_value='').tag(sync=True)
    icon = Union([Unicode(), Dict()]).tag(sync=True)
    width = Int(allow_none=True).tag(sync=True)
    label = Unicode(allow_none=True, default_value=None).tag(sync=True)


ButtonChoice(
    icon={
        'freeform': '<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg"><path stroke-width="2px" stroke="currentColor" fill="none" d="m15.99958,27.5687c-1.8418,-0.3359 -3.71385,-1.01143 -5.49959,-2.04243c-6.69178,-3.8635 -9.65985,-11.26864 -6.62435,-16.52628c3.0355,-5.25764 10.93258,-6.38978 17.62435,-2.52628c6.1635,3.5585 9.16819,10.12222 7.23126,15.24508"/><circle stroke-width="2px" stroke="currentColor" fill="none" r="3" cy="25" cx="27"/></svg>',
        'brush': '<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg"><path stroke-width="2px" stroke="currentColor" fill="none" d="m25.985,26.755c-5.345,2.455 -10.786,2.981 -14.455,1.899c-3.449,-1.017 -5.53,-3.338 -5.53,-6.654c0,-3.33 1.705,-4.929 3.835,-6.127c0.894,-0.503 1.88,-0.912 2.738,-1.451c0.786,-0.493 1.427,-1.143 1.427,-2.422c0,-1.692 -1.552,-2.769 -3.177,-3.649c-3.177,-1.722 -7.152,-2.378 -7.152,-2.378l0.658,-3.946c0,0 4.665,0.784 8.4,2.806c2.987,1.618 5.271,4.055 5.271,7.167c0,3.33 -1.705,4.929 -3.835,6.127c-0.894,0.503 -1.88,0.912 -2.738,1.451c-0.786,0.493 -1.427,1.143 -1.427,2.422c0,1.486 1.117,2.362 2.662,2.818c2.897,0.854 7.122,0.332 11.338,-1.551"/><circle stroke-width="2px" stroke="currentColor" fill="none" r="3" cy="24" cx="27"/></svg>',
        'rectangle': '<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg"><circle  stroke-width="2px" stroke="currentColor" fill="none" r="3" cy="24" cx="27"/><path stroke-linecap="square" stroke-width="2px" stroke="currentColor" fill="none" d="m24,24l-22,0l0,-19l25,0l0,16"/></svg>',
    },
    tooltip='Lasso Type',
    width=36,
    value='freeform',
    options={
        'freeform': 'Freeform',
        'brush': 'Brush',
        'rectangle': 'Rectangle',
    },
)
