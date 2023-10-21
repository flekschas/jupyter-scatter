import { createElementWithClass } from './utils';

const DEFAULT_BACKGROUND_COLOR = '#999999';
const DEFAULT_HIGHLIGHT_COLOR = '#000000';
const BIN_SPACE = 1;
const COMPARE = (new Intl.Collator(
  undefined,
  { numeric: true, sensitivity: 'base' }
)).compare;

const createCategoricalHistogramBackground = (canvas, data)  => {
  const ctx = canvas.getContext("2d");

  const state = {
    width: 100,
    height: 10,
    lastI: Object.values(data).length - 1,
    color: DEFAULT_BACKGROUND_COLOR,
  }

  const style = (newColor) => {
    state.color = newColor;
  }

  const draw = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = state.color;
    state.rects.forEach((rect) => {
      ctx.fillRect(rect.x, 0, rect.width, state.height);
    });
  }

  const resize = (width, height) => {
    state.width = width * window.devicePixelRatio;
    state.height = height * window.devicePixelRatio;
    init();
  }

  const init = () => {
    const { lastI, width } = state;
    let cumulativePercentage = 0;

    state.rects = Object.entries(data)
      .sort(([keyA], [keyB]) => COMPARE(keyA, [keyB]))
      .map(([, value], i) => {
        const rect = {
          x: cumulativePercentage * width,
          width: i === lastI ? value * width : (value * width) - 1,
        }
        cumulativePercentage += value;
        return rect;
      });
  }

  init();

  return { draw, style, resize };
}

const createCategoricalHistogramHighlight = (canvas, data)  => {
  const ctx = canvas.getContext("2d");

  const state = {
    width: 100,
    height: 10,
    lastI: Object.values(data).length - 1,
    color: DEFAULT_HIGHLIGHT_COLOR,
  }

  const style = (newColor) => {
    state.color = newColor;
  }

  const draw = (key) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = state.color;
    const rect = state.rects[key];
    ctx.fillRect(rect.x, 0, rect.width, state.height);
  }

  const resize = (width, height) => {
    state.width = width * window.devicePixelRatio;
    state.height = height * window.devicePixelRatio;
    init();
  }

  const init = () => {
    const { lastI, width } = state;
    let cumulativePercentage = 0;

    state.rects = Object.entries(data)
      .sort(([keyA], [keyB]) => COMPARE(keyA, [keyB]))
      .reduce((acc, [key, value], i) => {
        acc[key] = {
          x: cumulativePercentage * width,
          width: i === lastI ? value * width : (value * width) - 1,
        }
        cumulativePercentage += value;
        return acc;
      }, {});
  }

  init();

  return { draw, style, resize };
}

const createNumericalHistogramBackground = (canvas, data) => {
  const ctx = canvas.getContext("2d");

  const state = {
    width: 100,
    height: 10,
    color: DEFAULT_BACKGROUND_COLOR,
  }

  const style = (newColor) => {
    state.color = newColor;
  }

  const draw = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = state.color;
    state.rects.forEach((rect) => {
      ctx.fillRect(rect.x, rect.y, state.binWidth, rect.height);
    });
  }

  const resize = (width, height) => {
    state.width = width * window.devicePixelRatio;
    state.height = height * window.devicePixelRatio;
    init();
  }

  const init = () => {
    const { width, height } = state;

    state.binStep = width / data.length;
    state.binWidth = state.binStep - BIN_SPACE * window.devicePixelRatio;
    state.rects = data.map((value, i) => ({
      x: i * state.binStep,
      y: (1 - value) * height,
      height: value * height,
    }));
  }

  init();

  return { draw, style, resize };
}

const createNumericalHistogramHighlight = (canvas, data) => {
  const ctx = canvas.getContext("2d");

  const state = {
    width: 100,
    height: 10,
    color: DEFAULT_HIGHLIGHT_COLOR,
  }

  const style = (newColor) => {
    state.color = newColor;
  }

  const draw = (bin) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = state.color;
    const rect = state.rects[bin];
    ctx.fillRect(rect.x, rect.y, state.binWidth, rect.height);
  }

  const resize = (width, height) => {
    state.width = width * window.devicePixelRatio;
    state.height = height * window.devicePixelRatio;
    init();
  }

  const init = () => {
    const { width, height } = state;

    state.binStep = width / data.length;
    state.binWidth = state.binStep - BIN_SPACE * window.devicePixelRatio;
    state.rects = data.map((value, i) => ({
      x: i * state.binStep,
      y: (1 - value) * height,
      height: value * height,
    }));
  }

  init();

  return { draw, style, resize };
}

export const createHistogram = (width, height) => {
  const element = createElementWithClass('div', 'histogram');
  element.style.position = 'relative';
  element.style.width = width;
  element.style.height = height;

  const background = document.createElement('canvas');
  background.style.position = 'absolute';
  background.style.top = 0;
  background.style.left = 0;

  const foreground = document.createElement('canvas');
  foreground.style.position = 'absolute';
  foreground.style.top = 0;
  foreground.style.left = 0;

  element.appendChild(background);
  element.appendChild(foreground);

  let histogramBackground;
  let histogramHighlight;

  let isInit = false;
  let isBackgroundDrawn = false;

  const draw = (key) => {
    if (!isInit) return;
    if (!isBackgroundDrawn) {
      histogramBackground.draw();
      isBackgroundDrawn = true;
    }
    histogramHighlight.draw(key);
  }

  const style = (color, background) => {
    if (!isInit) return;
    histogramBackground.style(background);
    histogramHighlight.style(color);
  }

  const resize = () => {
    const bBox = element.getBoundingClientRect();

    const width = Math.round(bBox.width);
    const height = Math.round(bBox.height);

    background.width = width * window.devicePixelRatio;
    background.height = height * window.devicePixelRatio;
    background.style.width = `${width}px`;
    background.style.height = `${height}px`;
    foreground.width = width * window.devicePixelRatio;
    foreground.height = height * window.devicePixelRatio;
    foreground.style.width = `${width}px`;
    foreground.style.height = `${height}px`;

    if (!isInit) return;

    histogramBackground.resize(width, height);
    histogramHighlight.resize(width, height);
  }

  const init = (data, isCategorical) => {
    isInit = Boolean(data);

    if (!isInit) return;

    if (isCategorical) {
      histogramBackground = createCategoricalHistogramBackground(background, data);
      histogramHighlight = createCategoricalHistogramHighlight(foreground, data);
    } else {
      histogramBackground = createNumericalHistogramBackground(background, data);
      histogramHighlight = createNumericalHistogramHighlight(foreground, data);
    }

    window.requestAnimationFrame(() => {
      resize();
    });
  }

  return { element, init, draw, style, resize };
}
