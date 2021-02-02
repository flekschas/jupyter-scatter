const version = require('./package.json').version;
version.split('.')

const m = version.match(/(\d+).(\d+).(\d+)-?([a-z]+)?.?(\d+)?/)

let versionInfo = '(' + m[1] + ', ' + m[2] + ', ' + m[3];

versionInfo += m[4] !== undefined
  ? ', ' + m[4]
  : ', \'final\'';

versionInfo += m[5] !== undefined
  ? ', ' + m[5]
  : ', 1';

versionInfo += ')'

module.exports = versionInfo
