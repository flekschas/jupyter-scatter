import { getD3FormatSpecifier } from '@flekschas/utils';
import { format } from 'd3-format';

const sortOrder = {
  'color': 0,
  'opacity': 1,
  'size': 2,
  'connection_color': 3,
  'connection_opacity': 4,
  'connection_size': 5,
}

function createLabelFormatter(valueRange, isCategorical) {
  const min = valueRange[0];
  const max = valueRange[1];

  if (isCategorical || Number.isNaN(Number(min)) || Number.isNaN(Number(max))) {
    return function (value) { return value };
  }

  return format(getD3FormatSpecifier(valueRange));
}

function createValue(value) {
  const element = document.createElement('span');
  element.className = 'legend-value';
  element.style.marginLeft = '0.25rem';

  element.textContent = value;

  return element;
}

function createLabel(label) {
  const element = document.createElement('span');
  element.className = 'legend-label';
  element.style.opacity = 0.5;

  element.textContent = label || '';

  return element;
}

function createIcon(
  visualChannel,
  encoding,
  encodingRange,
  sizePx,
  fontColor
) {
  const element = document.createElement('div');
  element.className = 'legend-icon';
  element.style.width = sizePx + 'px';
  element.style.height = sizePx + 'px';
  element.style.borderRadius = sizePx + 'px';
  element.style.backgroundColor = 'rgb(' + fontColor + ','  + fontColor + ',' + fontColor + ')';

  if (visualChannel.includes('color')) {
    element.style.backgroundColor = Array.isArray(encoding)
      ? 'rgb(' + encoding.slice(0, 3).map((v) => v * 255).join(', ') + ')'
      : encoding;
  } else if (visualChannel.includes('opacity')) {
    element.style.backgroundColor = 'rgba(' + fontColor + ',' + fontColor + ','  + fontColor + ',' + encoding + ')';
    if (encoding < 0.2) {
      element.style.boxShadow = 'inset 0 0 1px rgba(' + fontColor + ',' + fontColor + ','  + fontColor + ', 0.66)';
    }
  } else if (visualChannel.includes('size')) {
    const minValue = Math.min.apply(null, encodingRange);
    const maxValue = Math.max.apply(null, encodingRange);
    const extent = maxValue - minValue;
    const normEncoding = 0.2 + ((encoding - minValue) / extent) * 0.8;
    element.style.transform = `scale(${normEncoding})`;
  }

  return element;
}

function createEntry(
  visualChannel,
  value,
  encodedValue,
  encodingRange,
  sizePx,
  fontColor
) {
  const element = document.createElement('div');
  element.className = 'legend-entry';
  element.style.display = 'flex';
  element.style.alignItems = 'center';

  element.appendChild(
    createIcon(visualChannel, encodedValue, encodingRange, sizePx, fontColor)
  );
  element.appendChild(createValue(value));

  return element;
}

function createTitle(visualChannel, isRightAligned) {
  const element = document.createElement('div');
  element.className = 'legend-title';
  element.style.textTransform = 'capitalize';
  element.style.fontWeight = 'bold';
  if (isRightAligned) element.style.textAlign = 'right';
  element.textContent = visualChannel
    .replace('connection', 'line')
    .replaceAll('_', ' ');

  return element;
}

function createEncoding() {
  const element = document.createElement('div');
  element.className = 'legend-encoding';
  element.style.display = 'grid';
  element.style.gridTemplateColumns = 'max-content max-content';
  element.style.gap = '0 0.2rem';
  element.style.height = 'min-content';

  return element;
}

export function createLegend(encodings, fontColor, backgroundColor, size) {
  const f = fontColor ? fontColor[0] * 255 : 0;
  const b = backgroundColor ? backgroundColor[0] * 255 : 255;

  let sizePx = 10;
  if (size === 'medium') sizePx = 12;
  else if (size === 'large') sizePx = 16;

  const root = document.createElement('div');
  root.className = 'legend';
  root.style.display = 'flex';
  root.style.gap = sizePx + 'px';
  root.style.margin = (sizePx * 0.2) + 'px';
  root.style.padding = (sizePx * 0.25) + 'px';
  root.style.fontSize = sizePx + 'px';
  root.style.borderRadius = (sizePx * 0.25) + 'px';
  root.style.color = 'rgb(' + f + ', ' + f + ', ' + f + ')';
  root.style.backgroundColor = 'rgba(' + b + ', ' + b + ', ' + b + ', 0.85)';
  root.style.pointerEvents = 'none';
  root.style.userSelect = 'none';

  Object.entries(encodings)
    .sort((a, b) => sortOrder[a[0]] - sortOrder[b[0]])
    .forEach((encodingEntry) => {
      const visualChannel = encodingEntry[0];
      const visualEncoding = encodingEntry[1];
      const encoding = createEncoding();
      encoding.appendChild(createTitle(visualChannel, Boolean(visualEncoding.variable)));
      encoding.appendChild(createLabel(visualEncoding.variable));

      const valueRange = [
        visualEncoding.values.at(0).at(0),
        visualEncoding.values.at(-1).at(0)
      ];

      const encodingRange = [
        visualEncoding.values.at(0).at(1),
        visualEncoding.values.at(-1).at(1)
      ];

      const formatter = createLabelFormatter(
        valueRange,
        visualEncoding.categorical
      );

      const values = typeof visualEncoding.values[0][0] === 'number'
        ? [...visualEncoding.values].reverse()
        : visualEncoding.values;

      values.forEach(([value, encodedValue, label]) => {
        encoding.appendChild(
          createEntry(
            visualChannel,
            formatter(value),
            encodedValue,
            encodingRange,
            sizePx,
            f
          )
        );
        encoding.appendChild(createLabel(label));
      });

      root.append(encoding);
    });

  return root;
}
