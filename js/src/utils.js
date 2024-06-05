import { scaleLog, scalePow, scaleLinear, scaleOrdinal, scaleTime } from 'd3-scale';
import { utcFormat } from 'd3-time-format';

export function camelToSnake(string) {
  return string.replace(/[\w]([A-Z])/g, (m) => m[0] + "_" + m[1]).toLowerCase();
}

export function toCapitalCase(string) {
  if (string.length === 0) return string;
  return string.at(0).toUpperCase() + string.slice(1);
}

export function toTitleCase(string) {
  return string.split(/[\s_]/).map((s) => toCapitalCase(s)).join(' ').split('-').map((s) => toCapitalCase(s)).join('-')
}

export function toHtmlClass(string) {
  return string
    // Lower case the string for simplicity
    .toLowerCase()
    // Replace any leading characters that are not a-z
    .replace(/^[^a-z]*/g, '')
    // Replace any white space with a hyphen
    .replace(/\s/g, '-')
    // Remove any character other than alphabetical, numerical, underscore, or hyphen
    .replace( /[^a-z0-9_-]/g, '');
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
    })
  );

  document.body.removeChild(link);
}

export function getScale(scaleType) {
  if (scaleType.startsWith('log'))
    return scaleLog().base(scaleType.split('_')[1] || 10);

  if (scaleType.startsWith('pow'))
    return scalePow().exponent(scaleType.split('_')[1] || 2);

  if (scaleType === 'time')
    return scaleTime();

  if (scaleType === 'linear')
    return scaleLinear();

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
  if (size === 'large') return '1rem';
  if (size === 'medium') return '0.85rem';
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
      className.forEach((name) => element.classList.add(name));
    } else {
      element.classList.add(className);
    }
  }

  return element;
}

export function remToPx(rem) {
  return rem * parseFloat(getComputedStyle(document.documentElement).fontSize);
}

const formatMillisecond = utcFormat("%a, %b %e, %Y %H:%M:%S.%L");
const formatSecond = utcFormat("%a, %b %e, %Y %H:%M:%S");
const formatMinute = utcFormat("%a, %b %e, %Y %H:%M");
const formatHour = utcFormat("%a, %b %e, %Y %H");
const formatDay = utcFormat("%a, %b %e, %Y");
const formatWeek = utcFormat("%b %e, %Y");
const formatMonth = utcFormat("%b %Y");
const formatYear = utcFormat("%Y");

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

  const intervals = []
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
  if (medianInterval > YEAR_MSEC) return formatYear;
  if (medianInterval > MONTH_MSEC) return formatMonth;
  if (medianInterval > WEEK_MSEC) return formatWeek;
  if (medianInterval > DAY_MSEC) return formatDay;
  if (medianInterval > HOUR_MSEC) return formatHour;
  if (medianInterval > MIN_MSEC) return formatMinute;
  if (medianInterval > SEC_MSEC) return formatSecond;
  return formatMillisecond;
}

export function createXTimeFormat(points) {
  return createTimeFormat(points, (point) => point[0])
}

export function createYTimeFormat(points) {
  return createTimeFormat(points, (point) => point[1])
}
