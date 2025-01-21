import anywidget

from traitlets import Dict, Int, Unicode


class ButtonIntSlider(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
      const button = document.createElement('button');
      const dialog = document.createElement('dialog');
      const container = document.createElement('div');
      const slider = document.createElement('input');
      const sliderLabel = document.createElement('label');
      const sliderLabelValue = document.createElement('span');
      const sliderLabelText = document.createElement('span');

      button.classList.add('jupyter-widgets', 'jupyter-button', 'widget-button', 'jupyter-scatter-slider-toggler');
      dialog.classList.add('jupyter-scatter-slider-dialog');
      container.classList.add('jupyter-widgets', 'jupyter-scatter-slider-container');
      slider.type = 'range';
      sliderLabel.classList.add('jupyter-scatter-slider-label');
      sliderLabelText.classList.add('jupyter-scatter-slider-label-text');

      let open = false;

      const update = () => {
        const value = model.get('value');
        const valueMin = model.get('value_min');
        const valueMax = model.get('value_max');
        const valueStep = model.get('value_step');
        const description = model.get('description');
        const icon = model.get('icon');
        const tooltip = model.get('tooltip');
        const width = model.get('width');
        const sliderWidth = model.get('slider_width');
        const _sliderLabelText = model.get('slider_label');
        const sliderLabelWidth = model.get('slider_label_width');
        const sliderLabelValueSuffix = model.get('slider_label_value_suffix');

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

        if (value !== undefined) {
          slider.value = value;
          sliderLabelValue.textContent = sliderLabelValueSuffix
            ? `${value}${sliderLabelValueSuffix}`
            : value;
        }

        if (valueMin !== undefined) {
          slider.min = valueMin;
        } else {
          slider.removeAttribute('min');
        }

        if (valueMax !== undefined) {
          slider.max = valueMax;
        } else {
          slider.removeAttribute('max');
        }

        if (valueStep !== undefined) {
          slider.step = valueStep;
        } else {
          slider.removeAttribute('step');
        }

        if (sliderWidth) {
          slider.style.width = `${sliderWidth}px`;
        } else {
          slider.style.removeProperty('width');
        }

        if (sliderLabelWidth) {
          sliderLabelValue.style.width = `${sliderLabelWidth}px`;
        } else {
          sliderLabelValue.style.removeProperty('width');
        }

        if (_sliderLabelText) {
          sliderLabelText.textContent = _sliderLabelText;
        } else {
          sliderLabelText.textContent = '';
        }
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
        const value = event.target.value;
        const suffix = model.get('slider_label_value_suffix');

        sliderLabelValue.textContent = suffix
          ? `${value}${suffix}`
          : value;

        model.set('value', Number(value));
        model.save_changes();
      }

      const inputHandler = (event) => {
        const value = event.target.value;
        const suffix = model.get('slider_label_value_suffix');

        sliderLabelValue.textContent = suffix
          ? `${value}${suffix}`
          : value;
      }

      button.addEventListener('click', clickHandler);
      slider.addEventListener('change', changeHandler);
      slider.addEventListener('input', inputHandler);

      model.on('change:description', update);
      model.on('change:icon', update);
      model.on('change:width', update);
      model.on('change:tooltip', update);
      model.on('change:value', update);
      model.on('change:value_min', update);
      model.on('change:value_max', update);
      model.on('change:value_step', update);
      model.on('change:slider_width', update);
      model.on('change:slider_label', update);
      model.on('change:slider_label_width', update);
      model.on('change:slider_label_value_suffix', update);

      update();

      sliderLabel.appendChild(sliderLabelValue);
      sliderLabel.appendChild(sliderLabelText);
      container.appendChild(slider);
      container.appendChild(sliderLabel);
      dialog.appendChild(container);

      el.appendChild(button);
      el.appendChild(dialog);

      return () => {
        button.removeEventListener('click', clickHandler);
        slider.removeEventListener('change', changeHandler);
        slider.removeEventListener('input', inputHandler);
        dialog.removeEventListener('click', dialogClickHandler);
      };
    }
    export default { render }
    """

    _css = """
    .jupyter-scatter-slider-toggler {
      position: relative;
    }
    .jupyter-scatter-slider-toggler > svg {
      display: block;
      width: 100%;
      height: 100%;
    }
    .jupyter-scatter-slider-dialog {
      position: absolute;
      padding: 0;
      margin: 0;
      border: none;
    }
    .jupyter-scatter-slider-container {
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
    .jupyter-scatter-slider-label {
      display: flex;
      align-items: center;
      gap: 0 0.5rem;
      white-space: nowrap;
    }
    ::backdrop{
      background: var(--jp-layout-color0);
      opacity: 0.33;
    }
    """

    value = Int().tag(sync=True)
    value_min = Int().tag(sync=True)
    value_max = Int().tag(sync=True)
    value_step = Int().tag(sync=True)
    description = Unicode(default_value='').tag(sync=True)
    icon = Unicode().tag(sync=True)
    width = Int(allow_none=True).tag(sync=True)
    slider_width = Int(allow_none=True).tag(sync=True)
    slider_label = Unicode(allow_none=True, default_value=None).tag(sync=True)
    slider_label_width = Int(allow_none=True, default_value=None).tag(sync=True)
    slider_label_value_suffix = Unicode(allow_none=True, default_value=None).tag(
        sync=True
    )
