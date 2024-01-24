import { scaleLog, scalePow, scaleLinear, scaleOrdinal } from 'd3-scale';

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

export function createNumericalBinGetter(histogram, domain, range) {
  const maxBinId = histogram.length - 1;
  const min = range ? range[0] : domain[0];
  const extent = range ? range[1] - range[0] : domain[1] - domain[0];
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
