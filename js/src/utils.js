import { scaleLog, scalePow, scaleLinear, scaleOrdinal } from 'd3-scale';

export function camelToSnake(string) {
  return string.replace(/[\w]([A-Z])/g, (m) => m[0] + "_" + m[1]).toLowerCase();
}

export function toCapitalCase(string) {
  if (string.length === 0) return string;
  return string.at(0).toUpperCase() + string.slice(1).toLowerCase();
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
