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

function createIcon(title, encoding, encodingRange) {
  const element = document.createElement('div');
  element.className = 'legend-icon';
  element.style.width = '0.625rem';
  element.style.height = '0.625rem';
  element.style.borderRadius = '0.625rem';
  element.style.backgroundColor = 'black';

  if (title === 'color') {
    element.style.backgroundColor = Array.isArray(encoding)
      ? 'rgb(' + encoding.slice(0, 3).map((v) => v * 255).join(', ') + ')'
      : encoding;
  } else if (title === 'opacity') {
    element.style.backgroundColor = 'rgba(0, 0, 0, ' + encoding + ')';
    if (encoding < 0.2) {
      element.style.boxShadow = 'inset 0 0 1px rgba(0, 0, 0, 0.5)';
    }
  } else if (title === 'size') {
    const extent = encodingRange[1] - encodingRange[0];
    const normEncoding = 0.2 + ((encoding - encodingRange[0]) / extent) * 0.8;
    element.style.transform = `scale(${normEncoding})`;
  }

  return element;
}

function createEntry(title, value, encoding, encodingRange) {
  const element = document.createElement('div');
  element.className = 'legend-entry';
  element.style.display = 'flex';
  element.style.alignItems = 'center';

  element.appendChild(createIcon(title, encoding, encodingRange));
  element.appendChild(createLabel(value));

  return element;
}

function createTitle(title) {
  const element = document.createElement('div');
  element.className = 'legend-title';
  element.style.fontWeight = 'bold';

  element.textContent = title[0].toUpperCase() + title.slice(1);

  return element;
}

function createEncoding() {
  const element = document.createElement('div');
  element.className = 'legend-encoding';
  element.style.display = 'flex';
  element.style.flexDirection = 'column';

  return element;
}

function createLegend(encodings) {
  const root = document.createElement('div');
  root.className = 'legend';
  root.style.display = 'flex';
  root.style.gap = '0.5rem';
  root.style.margin = '2px';
  root.style.padding = '0.25rem';
  root.style.fontSize = '0.625rem';
  root.style.borderRadius = '0.25rem';
  root.style.backgroundColor = 'rgba(255, 255, 255, 0.85)';
  root.style.pointerEvents = 'none';
  root.style.userSelect = 'none';

  Object.entries(encodings).forEach(([title, valueEncodingPairs]) => {
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
        createEntry(title, formatter(value), encodedValue, encodingRange)
      );
    });

    root.append(encoding);
  });

  return root;
}

module.exports = createLegend;
