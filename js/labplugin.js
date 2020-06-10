var jupyterScatterplot = require('./index');
var base = require('@jupyter-widgets/base');

module.exports = {
  id: 'jupyter.extensions.jupyter-scatterplot',
  requires: [base.IJupyterWidgetRegistry],
  activate: function(app, widgets) {
      widgets.registerWidget({
          name: 'jupyter-scatterplot',
          version: jupyterScatterplot.version,
          exports: jupyterScatterplot
      });
  },
  autoStart: true
};