const sortOrder = {
  'color': 0,
  'opacity': 1,
  'size': 2,
  'connection_color': 3,
  'connection_opacity': 4,
  'connection_size': 5,
}

function createLabelFormatter(valueRange) {
  const min = valueRange[0];
  const max = valueRange[1];

  if (Number.isNaN(Number(min)) || Number.isNaN(Number(max))) {
    return function (value) { return value };
  }

  const extent = max - min;

  const i = Math.floor(Math.log10(extent));
  const k = Math.max(0, i >= 0 ? 2 - i : 1 - i);
  const l = Math.pow(10, k);

  return function (value) { return (Math.round(value * l) / l).toFixed(k); }
}

function createLabel(value) {
  const element = document.createElement('span');
  element.className = 'legend-label';
  element.style.marginLeft = '0.25rem';

  element.textContent = value;

  return element;
}

function createIcon(title, encoding, encodingRange, sizePx, fontColor) {
  const element = document.createElement('div');
  element.className = 'legend-icon';
  element.style.width = sizePx + 'px';
  element.style.height = sizePx + 'px';
  element.style.borderRadius = sizePx + 'px';
  element.style.backgroundColor = 'rgb(' + fontColor + ','  + fontColor + ',' + fontColor + ')';

  if (title.indexOf('color') >= 0) {
    element.style.backgroundColor = Array.isArray(encoding)
      ? 'rgb(' + encoding.slice(0, 3).map((v) => v * 255).join(', ') + ')'
      : encoding;
  } else if (title.indexOf('opacity') >= 0) {
    element.style.backgroundColor = 'rgba(' + fontColor + ',' + fontColor + ','  + fontColor + ',' + encoding + ')';
    if (encoding < 0.2) {
      element.style.boxShadow = 'inset 0 0 1px rgba(' + fontColor + ',' + fontColor + ','  + fontColor + ', 0.66)';
    }
  } else if (title.indexOf('size') >= 0) {
    const extent = encodingRange[1] - encodingRange[0];
    const normEncoding = 0.2 + ((encoding - encodingRange[0]) / extent) * 0.8;
    element.style.transform = `scale(${normEncoding})`;
  }

  return element;
}

function createEntry(title, value, encoding, encodingRange, sizePx, fontColor) {
  const element = document.createElement('div');
  element.className = 'legend-entry';
  element.style.display = 'flex';
  element.style.alignItems = 'center';

  element.appendChild(createIcon(title, encoding, encodingRange, sizePx, fontColor));
  element.appendChild(createLabel(value));

  return element;
}

function createTitle(title) {
  const element = document.createElement('div');
  element.className = 'legend-title';
  element.style.fontWeight = 'bold';
  element.style.textTransform = 'capitalize';

  element.textContent = title.replace('connection', 'line').replaceAll('_', ' ');

  return element;
}

function createEncoding() {
  const element = document.createElement('div');
  element.className = 'legend-encoding';
  element.style.display = 'flex';
  element.style.flexDirection = 'column';

  return element;
}

function createLegend(encodings, fontColor, backgroundColor, size) {
  const f = fontColor ? fontColor[0] * 255 : 0;
  const b = backgroundColor ? backgroundColor[0] * 255 : 255;

  let sizePx = 10;
  if (size === 'medium') sizePx = 12;
  else if (size === 'large') sizePx = 16;

  const root = document.createElement('div');
  root.className = 'legend';
  root.style.display = 'flex';
  root.style.gap = (sizePx * 0.5) + 'px';
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
      const title = encodingEntry[0];
      const valueEncodingPairs = encodingEntry[1];
      const encoding = createEncoding();
      encoding.appendChild(createTitle(title));

      const valueRange = [
        valueEncodingPairs[0][0],
        valueEncodingPairs[valueEncodingPairs.length - 1][0]
      ];

      const encodingRange = [
        valueEncodingPairs[0][1],
        valueEncodingPairs[valueEncodingPairs.length - 1][1]
      ];

      const formatter = createLabelFormatter(valueRange);

      valueEncodingPairs.forEach(([value, encodedValue]) => {
        encoding.appendChild(
          createEntry(
            title,
            formatter(value),
            encodedValue,
            encodingRange,
            sizePx,
            f
          )
        );
      });

      root.append(encoding);
    });

  return root;
}

module.exports = createLegend;
