// Export widget models and views, and the npm package version number.
module.exports = require('./src/index.js');
module.exports['version'] = require('./package.json').version;
