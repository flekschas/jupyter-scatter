const plugin = require('./index');
const base = require('@jupyter-widgets/base');
const widgetName = require('./package.json').name;

module.exports = {
  id: 'jupyter.extensions.' + widgetName,
  requires: [base.IJupyterWidgetRegistry],
  activate: function(app, widgets) {
    widgets.registerWidget({
      name: widgetName,
      version: plugin.version,
      exports: plugin
    });
  },
  autoStart: true
};
