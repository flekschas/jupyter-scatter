const plugin = require('./index');
const base = require('@jupyter-widgets/base');
const name = require('./package.json').name;

module.exports = {
  id: 'jupyter.extensions.' + name,
  requires: [base.IJupyterWidgetRegistry],
  activate: function(app, widgets) {
    widgets.registerWidget({
      name: name,
      version: plugin.version,
      exports: plugin
    });
  },
  autoStart: true
};
