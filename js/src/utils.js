import {
  scaleLinear,
  scaleLog,
  scaleOrdinal,
  scalePow,
  scaleTime,
} from 'd3-scale';
import { utcFormat } from 'd3-time-format';

const camelToSnakeRegex = /[\w]([A-Z])/g;
export function camelToSnake(string) {
  return string
    .replace(camelToSnakeRegex, (m) => `${m[0]}_${m[1]}`)
    .toLowerCase();
}

const snakeToCamelRegex = /[-_][a-z]/g;
export function snakeToCamel(string) {
  return string
    .toLowerCase()
    .replace(snakeToCamelRegex, (group) => group.slice(-1).toUpperCase());
}

export function toCapitalCase(string) {
  if (string.length === 0) {
    return string;
  }
  return string.at(0).toUpperCase() + string.slice(1);
}

const titleCaseRegex = /[\s_]/;
export function toTitleCase(string) {
  return string
    .split(titleCaseRegex)
    .map((s) => toCapitalCase(s))
    .join(' ')
    .split('-')
    .map((s) => toCapitalCase(s))
    .join('-');
}

export function toHtmlClass(string) {
  return (
    string
      // Lower case the string for simplicity
      .toLowerCase()
      // Replace any leading characters that are not a-z
      .replace(/^[^a-z]*/g, '')
      // Replace any white space with a hyphen
      .replace(/\s/g, '-')
      // Remove any character other than alphabetical, numerical, underscore, or hyphen
      .replace(/[^a-z0-9_-]/g, '')
  );
}

export function downloadBlob(blob, name) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = name || 'jscatter.png';

  document.body.appendChild(link);

  link.dispatchEvent(
    new MouseEvent('click', {
      bubbles: true,
      cancelable: true,
      view: window,
    }),
  );

  document.body.removeChild(link);
}

export function getScale(scaleType) {
  if (scaleType.startsWith('log')) {
    return scaleLog().base(scaleType.split('_')[1] || 10);
  }

  if (scaleType.startsWith('pow')) {
    return scalePow().exponent(scaleType.split('_')[1] || 2);
  }

  if (scaleType === 'time') {
    return scaleTime();
  }

  if (scaleType === 'linear') {
    return scaleLinear();
  }

  return scaleOrdinal();
}

export function invertObjToMap(obj) {
  return Object.keys(obj).reduce((invertedMap, key) => {
    invertedMap.set(obj[key], key);
    return invertedMap;
  }, new Map());
}

export function createOrdinalScaleInverter(domain) {
  const invertedDomainMap = invertObjToMap(domain);
  return (value) => invertedDomainMap.get(value);
}

export function getTooltipFontSize(size) {
  if (size === 'large') {
    return '1rem';
  }
  if (size === 'medium') {
    return '0.85rem';
  }
  return '0.675rem';
}

export function createNumericalBinGetter(histogram, domain) {
  const maxBinId = histogram.length - 1;
  const min = domain[0];
  const extent = domain[1] - domain[0];
  return (value) => Math.round(((value - min) / extent) * maxBinId);
}

export function createElementWithClass(tagName, className) {
  const element = document.createElement(tagName);

  if (className) {
    if (Array.isArray(className)) {
      for (const name of className) {
        element.classList.add(name);
      }
    } else {
      element.classList.add(className);
    }
  }

  return element;
}

export function remToPx(rem) {
  return (
    rem * Number.parseFloat(getComputedStyle(document.documentElement).fontSize)
  );
}

const formatMillisecond = utcFormat('%a, %b %e, %Y %H:%M:%S.%L');
const formatSecond = utcFormat('%a, %b %e, %Y %H:%M:%S');
const formatMinute = utcFormat('%a, %b %e, %Y %H:%M');
const formatHour = utcFormat('%a, %b %e, %Y %H');
const formatDay = utcFormat('%a, %b %e, %Y');
const formatWeek = utcFormat('%b %e, %Y');
const formatMonth = utcFormat('%b %Y');
const formatYear = utcFormat('%Y');

const SEC_MSEC = 1000;
const MIN_MSEC = 60 * SEC_MSEC;
const HOUR_MSEC = 60 * MIN_MSEC;
const DAY_MSEC = 24 * HOUR_MSEC;
const WEEK_MSEC = 7 * DAY_MSEC;
const MONTH_MSEC = 4 * WEEK_MSEC;
const YEAR_MSEC = 365 * DAY_MSEC;

export function medianTimeInterval(points, accessor) {
  const values = [];
  for (const point of points) {
    values.push(accessor(point));
  }
  values.sort();

  const intervals = [];
  for (let i = 1; i < values.length; i++) {
    const interval = values[i] - values[i - 1];
    if (interval > 0) {
      intervals.push(interval);
    }
  }
  intervals.sort();

  return intervals[Math.floor(intervals.length / 2)];
}

export function createTimeFormat(points, accessor) {
  const medianInterval = medianTimeInterval(points, accessor);
  if (medianInterval > YEAR_MSEC) {
    return formatYear;
  }
  if (medianInterval > MONTH_MSEC) {
    return formatMonth;
  }
  if (medianInterval > WEEK_MSEC) {
    return formatWeek;
  }
  if (medianInterval > DAY_MSEC) {
    return formatDay;
  }
  if (medianInterval > HOUR_MSEC) {
    return formatHour;
  }
  if (medianInterval > MIN_MSEC) {
    return formatMinute;
  }
  if (medianInterval > SEC_MSEC) {
    return formatSecond;
  }
  return formatMillisecond;
}

// biome-ignore lint/style/useNamingConvention: XTime stands for x time
export function createXTimeFormat(points) {
  return createTimeFormat(points, (point) => point[0]);
}

// biome-ignore lint/style/useNamingConvention: YTime stands for y time
export function createYTimeFormat(points) {
  return createTimeFormat(points, (point) => point[1]);
}

const toRgbaElement = document.createElement('div');
export function toRgba(color) {
  if (Array.isArray) {
    if (color.length >= 4) {
      return color.map((c) => c * 255);
    }
    if (color.length === 3) {
      return [...color.map((c) => c * 255), 255];
    }
    return [0, 0, 0, 0];
  }
  toRgbaElement.style.backgroundColor = color;
  document.body.appendChild(toRgbaElement);
  const rgba = getComputedStyle(toRgbaElement)['background-color'];
  document.body.removeChild(toRgbaElement);
  return rgba
    .slice(rgba.indexOf('(') + 1, rgba.indexOf(')'))
    .split(',')
    .map(Number);
}

export function blend(imageDataBack, imageDataFront) {
  // Verify dimensions match
  if (
    imageDataBack.width !== imageDataFront.width ||
    imageDataBack.height !== imageDataFront.height
  ) {
    throw new Error('Image dimensions must match for blending');
  }

  const width = imageDataBack.width;
  const height = imageDataBack.height;
  const newData = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < imageDataBack.data.length; i += 4) {
    const bgR = imageDataBack.data[i];
    const bgG = imageDataBack.data[i + 1];
    const bgB = imageDataBack.data[i + 2];
    const bgA = imageDataBack.data[i + 3] / 255;

    const fgR = imageDataFront.data[i];
    const fgG = imageDataFront.data[i + 1];
    const fgB = imageDataFront.data[i + 2];
    const fgA = imageDataFront.data[i + 3] / 255;

    const outA = fgA + bgA * (1 - fgA);

    newData[i] = Math.round((fgR * fgA + bgR * bgA * (1 - fgA)) / outA);
    newData[i + 1] = Math.round((fgG * fgA + bgG * bgA * (1 - fgA)) / outA);
    newData[i + 2] = Math.round((fgB * fgA + bgB * bgA * (1 - fgA)) / outA);
    newData[i + 3] = Math.round(outA * 255);
  }

  return new ImageData(newData, width, height);
}

export function addBackgroundColor(imageData, backgroundColor) {
  const newData = new Uint8ClampedArray(imageData.width * imageData.height * 4);

  const bg = toRgba(backgroundColor);

  for (let i = 0; i < newData.length; i += 4) {
    const bgAlpha = 1 - imageData.data[i + 3] / 255;
    const fgAlpha = imageData.data[i + 3] / 255;
    newData[i] = bg[0] * bgAlpha + imageData.data[i] * fgAlpha;
    newData[i + 1] = bg[1] * bgAlpha + imageData.data[i + 1] * fgAlpha;
    newData[i + 2] = bg[2] * bgAlpha + imageData.data[i + 2] * fgAlpha;
    newData[i + 3] = 255;
  }

  return new ImageData(newData, imageData.width, imageData.height);
}

export function imageDataToCanvas(imageData) {
  const canvas = document.createElement('canvas');
  canvas.width = imageData.width;
  canvas.height = imageData.height;

  const ctx = canvas.getContext('2d');
  ctx.putImageData(imageData, 0, 0);

  return canvas;
}

function zoomScaleToZoomLevel(zoomScale) {
  if (zoomScale <= 1) {
    return 0;
  }
  return Math.floor(Math.log2(zoomScale));
}

function zoomLevelToZoomScale(zoomLevel) {
  return 2 ** zoomLevel;
}

export function createViewToTiles(
  xDomain,
  yDomain,
  baseZoomScale,
  maxZoomLevel = Number.POSITIVE_INFINITY,
) {
  const [xMin, xMax] = xDomain;
  const xExtent = xMax - xMin;

  const [yMin, yMax] = yDomain;
  const yExtent = yMax - yMin;

  function toTileZoomLevel(zoomScale) {
    return Math.min(maxZoomLevel, zoomScaleToZoomLevel(zoomScale));
  }

  function toTileX(x, tileZoomScale) {
    return Math.floor((x - xMin) / (xExtent / tileZoomScale));
  }

  function toTileY(y, tileZoomScale) {
    return Math.floor((y - yMin) / (yExtent / tileZoomScale));
  }

  return function viewToTiles(xView, yView, zoomScaleView) {
    const zoomScale = baseZoomScale * zoomScaleView;
    const tileZoomLevel = toTileZoomLevel(zoomScale);
    const tileZoomScale = zoomLevelToZoomScale(tileZoomLevel);
    const [tileXMin, tileXMax] = xView.map((x) => toTileX(x, tileZoomScale));
    const [tileYMin, tileYMax] = yView.map((y) => toTileY(y, tileZoomScale));

    const tiles = [];

    for (let y = tileYMin; y <= tileYMax; y++) {
      for (let x = tileXMin; x <= tileXMax; x++) {
        tiles.push(`${x},${y},${tileZoomLevel}`);
      }
    }

    return tiles;
  };
}
