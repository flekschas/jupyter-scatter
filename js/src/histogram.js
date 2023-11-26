import { hierarchy, treemap, treemapBinary, treemapDice } from 'd3-hierarchy';

import { createElementWithClass } from './utils';

const DEFAULT_BACKGROUND_COLOR = 'rgb(153, 153, 153)';
const DEFAULT_HIGHLIGHT_COLOR = 'rgb(0, 0, 0)';
const BIN_SPACE = 1;
const BORDER_WIDTH = 2;
const VERMILION = '#D55E00'; // A red color from Okabe Ito's color palette

const createTreemap = (data, width, height) => {
  const dpr = window.devicePixelRatio;
  const padding = BORDER_WIDTH * dpr + dpr;

  const tiling = Object.keys(data).length > 10
    ? treemapBinary
    : treemapDice;

  return (
    treemap()
      .tile(tiling)
      .size([width - padding * 2, height])
      .padding(dpr)
      .round(true)
  )(
    hierarchy({
      key: '__root__',
      children: Object.entries(data).map(([key, value]) => ({ key, value }))
    })
      .sum((d) => d.value)
      .sort((a, b) => b.value - a.value)
  );
}

const createCategoricalHistogramBackground = (canvas, data)  => {
  const ctx = canvas.getContext("2d");
  const lastI = data.length - 1;

  const state = {
    width: 100,
    height: 10,
    lastI,
    color: DEFAULT_BACKGROUND_COLOR,
  }

  const style = (newColor) => {
    state.color = newColor;
  }

  const draw = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = state.color;
    state.rects.forEach((rect) => {
      ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
    });
  }

  const resize = (width, height) => {
    state.width = width * window.devicePixelRatio;
    state.height = height * window.devicePixelRatio;
    init();
  }

  const init = () => {
    const dpr = window.devicePixelRatio;
    const padding = BORDER_WIDTH * dpr + dpr;
    const tree = createTreemap(data, state.width, state.height);

    state.rects = tree.leaves().map((leaf) => ({
      x: leaf.x0 + padding,
      y: leaf.y0,
      width: leaf.x1 - leaf.x0,
      height: leaf.y1 - leaf.y0
    }));
  }

  init();

  return { draw, style, resize };
}

const createCategoricalHistogramHighlight = (canvas, data)  => {
  const ctx = canvas.getContext("2d");
  const lastI = Object.values(data).length - 1;

  const state = {
    width: 100,
    height: 10,
    lastI,
    color: DEFAULT_HIGHLIGHT_COLOR,
  }

  const style = (newColor) => {
    state.color = newColor;
  }

  const draw = (key) => {
    const rect = state.rects[key];

    if (rect === undefined) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = state.color;
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
  }

  const resize = (width, height) => {
    state.width = width * window.devicePixelRatio;
    state.height = height * window.devicePixelRatio;
    init();
  }

  const init = () => {
    const dpr = window.devicePixelRatio;
    const padding = BORDER_WIDTH * dpr + dpr;
    const tree = createTreemap(data, state.width, state.height);

    state.rects = tree.leaves().reduce((acc, leaf) => {
      acc[leaf.data.key] = {
        x: leaf.x0 + padding,
        y: leaf.y0,
        width: leaf.x1 - leaf.x0,
        height: leaf.y1 - leaf.y0
      }
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
    const dpr = window.devicePixelRatio;
    const padding = BORDER_WIDTH * dpr + dpr;
    const histogramWidth = width - padding * 2;

    state.binStep = histogramWidth / data.length;
    state.binWidth = state.binStep - BIN_SPACE * dpr;
    state.rects = data.map((value, i) => ({
      x: padding + i * state.binStep,
      y: (1 - value) * height,
      height: value * height,
    }));
  }

  init();

  return { draw, style, resize };
}

const createNumericalHistogramHighlight = (canvas, data) => {
  const ctx = canvas.getContext("2d");
  const lastI = Object.values(data).length - 1;

  const toBg = (color) => `rgb(${color.match((/\d+/g)).join(', ')}, 0.1)`;

  const state = {
    width: 100,
    height: 10,
    color: DEFAULT_HIGHLIGHT_COLOR,
    bgColor: toBg(DEFAULT_HIGHLIGHT_COLOR),
  }

  const style = (newColor) => {
    state.color = newColor;
    state.bgColor = toBg(newColor);
  }

  const draw = (bin) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const dpr = window.devicePixelRatio;

    if (bin < 0) {
      ctx.fillStyle = VERMILION;
      ctx.fillRect(0, 0, state.borderWidth, dpr);
      ctx.fillRect(state.borderWidth - dpr, 0, dpr, state.height);
      ctx.fillRect(0, state.height - dpr, state.borderWidth, dpr);
    } else if (bin > lastI) {
      ctx.fillStyle = VERMILION;
      const x = state.width - state.borderWidth;
      ctx.fillRect(x, 0, state.borderWidth, dpr);
      ctx.fillRect(x, 0, dpr, state.height);
      ctx.fillRect(x, state.height - dpr, state.borderWidth, dpr);
    } else {
      const rect = state.rects[bin];

      ctx.lineWidth = dpr;
      ctx.strokeStyle = state.bgColor
      ctx.strokeRect(rect.x, 0, state.binWidth, state.height);

      ctx.fillStyle = state.color;
      ctx.fillRect(rect.x, rect.y, state.binWidth, rect.height);
    }
  }

  const resize = (width, height) => {
    state.width = width * window.devicePixelRatio;
    state.height = height * window.devicePixelRatio;
    init();
  }

  const init = () => {
    const { width, height } = state;
    const dpr = window.devicePixelRatio;

    state.borderWidth = BORDER_WIDTH * dpr;

    const padding = state.borderWidth + dpr;
    const histogramWidth = width - padding * 2;
    const binStep = histogramWidth / data.length;

    state.binWidth = binStep - BIN_SPACE * dpr;
    state.rects = data.map((value, i) => ({
      x: padding + i * binStep,
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
