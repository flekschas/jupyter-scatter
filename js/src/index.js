/* eslint-env browser */
const widgets = require('@jupyter-widgets/base');
const _ = require('lodash');
const reglScatterplot = require('regl-scatterplot/dist/regl-scatterplot.js');
const packageJson = require('../package.json');

const createScatterplot = reglScatterplot.default;

const JScatterModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(
    _.result(this, 'widgets.DOMWidgetModel.prototype.defaults'),
    {
      _model_name : 'JScatterModel',
      _model_module : packageJson.name,
      _model_module_version : packageJson.version,
      _view_name : 'JScatterView',
      _view_module : packageJson.name,
      _view_module_version : packageJson.version
    }
  )
});

function camelToSnake(string) {
  return string.replace(/[\w]([A-Z])/g, function(m) {
    return m[0] + "_" + m[1];
  }).toLowerCase();
}

const MIN_WIDTH = 240;

const properties = [
  'colorBy',
  'points',
  'selectedPoints',
  'height',
  'backgroundColor',
  'backgroundImage',
  'lassoColor',
  'lassoMinDelay',
  'lassoMinDist',
  'pointColor',
  'pointColorActive',
  'pointColorHover',
  'pointOpacity',
  'pointSize',
  'pointSizeSelected',
  'pointOutlineWidth',
  'showRecticle',
  'recticleColor',
  'cameraTarget',
  'cameraDistance',
  'cameraRotation',
  'cameraView',
  'otherOptions',
  'lassoInitiator',
  'mouseMode',
  'viewReset'
];

// Custom View. Renders the widget model.
const JScatterView = widgets.DOMWidgetView.extend({
  render: function render() {
    var self = this;

    properties.forEach(function(propertyName) {
      self[propertyName] = self.model.get(camelToSnake(propertyName));
    });

    this.height = this.model.get('height');

    // Create a random 6-letter string
    // From https://gist.github.com/6174/6062387
    var randomStr = (
      Math.random().toString(36).substring(2, 5) +
      Math.random().toString(36).substring(2, 5)
    );
    this.model.set('dom_element_id', randomStr);

    this.container = document.createElement('div');
    this.container.setAttribute('id', randomStr);
    this.container.style.position = 'relative'
    this.container.style.border = this.otherOptions.theme === 'dark'
      ? '#333333' : '#dddddd';
    this.container.style.borderRadius = '2px';
    this.container.style.height = this.height + 'px';

    this.el.appendChild(this.container);

    this.canvas = document.createElement('canvas');
    this.canvas.style.position = 'absolute';
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';

    this.container.appendChild(this.canvas);

    window.requestAnimationFrame(function init() {
      self.width = Math.max(MIN_WIDTH, self.el.getBoundingClientRect().width);

      const initialOptions = {
        canvas: self.canvas,
        width: self.width,
      }

      properties.forEach(function(propertyName) {
        initialOptions[propertyName] = self[propertyName];
      });

      self.scatterplot = createScatterplot(initialOptions);

      // eslint-disable-next-line
      console.log(
        'jscatter v' + packageJson.version +
        ' with regl-scatterplot v' + self.scatterplot.get('version')
      );

      self.container.api = self.scatterplot;

      // Listen to events from the JavaScript world
      self.pointoverHandlerBound = self.pointoverHandler.bind(self);
      self.pointoutHandlerBound = self.pointoutHandler.bind(self);
      self.selectHandlerBound = self.selectHandler.bind(self);
      self.deselectHandlerBound = self.deselectHandler.bind(self);
      self.scatterplot.subscribe('pointover', self.pointoverHandlerBound);
      self.scatterplot.subscribe('pointout', self.pointoutHandlerBound);
      self.scatterplot.subscribe('select', self.selectHandlerBound);
      self.scatterplot.subscribe('deselect', self.deselectHandlerBound);

      // Listen to messages from the Python world
      properties.forEach(function(propertyName) {
        self.model.on(
          'change:' + camelToSnake(propertyName),
          self.withModelChangeHandler(
            propertyName,
            self[propertyName + 'Handler'].bind(self)
          ),
          self
        );
      });

      window.addEventListener('resize', self.resizeHandler.bind(self));
      window.addEventListener('deviceorientation', self.resizeHandler.bind(self));

      self.resizeHandler();
      self.colorCanvas();

      if (self.points.length) {
        self.scatterplot
          .draw(self.points)
          .then(function onInitialDraw() {
            if (self.selectedPoints.length) {
              self.scatterplot.select(self.selectedPoints);
            }
          });
      }
    });

    this.model.save_changes();
  },

  // Helper
  colorCanvas: function colorCanvas() {
    if (Array.isArray(this.backgroundColor)) {
      this.canvas.style.backgroundColor = 'rgb(' +
        this.backgroundColor.slice(0, 3).map(function (x) { return x * 255 }).join(',') +
        ')';
    } else {
      this.canvas.style.backgroundColor = this.backgroundColor;
    }
  },

  // Event handlers for JS-triggered events
  pointoverHandler: function pointoverHandler(pointIndex) {
    this.model.set('hovered_point', pointIndex);
    this.model.save_changes();
  },

  pointoutHandler: function pointoutHandler() {
    this.model.set('hovered_point', null);
    this.model.save_changes();
  },

  selectHandler: function selectHandler(event) {
    if (this.selectedPointsChangedPython) {
      this.selectedPointsChangedPython = false;
      return;
    }
    this.model.set('selected_points', event.points);
    this.selectedPointsChanged = true;
    this.model.save_changes();
  },

  deselectHandler: function deselectHandler() {
    if (this.selectedPointsChangedPython) {
      this.selectedPointsChangedPython = false;
      return;
    }
    this.model.set('selected_points', []);
    this.selectedPointsChanged = true;
    this.model.save_changes();
  },

  // Event handlers for Python-triggered events
  pointsHandler: function pointsHandler(newPoints) {
    this.scatterplot.draw(newPoints);
  },

  selectedPointsHandler: function selectedPointsHandler(newSelectedPoints) {
    this.selectedPointsChangedPython = true;
    if (!newSelectedPoints || !newSelectedPoints.length) {
      this.scatterplot.deselect({ preventEvent: true });
    } else {
      this.scatterplot.select(newSelectedPoints, { preventEvent: true });
    }
  },

  heightHandler: function heightHandler(newValue) {
    this.withPropertyChangeHandler('height', newValue);
    this.resizeHandler();
  },

  backgroundColorHandler: function backgroundColorHandler(newValue) {
    this.withPropertyChangeHandler('backgroundColor', newValue);
    this.colorCanvas();
  },

  backgroundImageHandler: function backgroundImageHandler(newValue) {
    this.withPropertyChangeHandler('backgroundImage', newValue);
  },

  colorByHandler: function colorByHandler(newValue) {
    this.withPropertyChangeHandler('colorBy', newValue);
  },

  lassoColorHandler: function lassoColorHandler(newValue) {
    this.withPropertyChangeHandler('lassoColor', newValue);
  },

  lassoMinDelayHandler: function lassoMinDelayHandler(newValue) {
    this.withPropertyChangeHandler('lassoMinDelay', newValue);
  },

  lassoMinDistHandler: function lassoMinDistHandler(newValue) {
    this.withPropertyChangeHandler('lassoMinDist', newValue);
  },

  pointColorHandler: function pointColorHandler(newValue) {
    this.withPropertyChangeHandler('pointColor', newValue);
  },

  pointColorActiveHandler: function pointColorActiveHandler(newValue) {
    this.withPropertyChangeHandler('pointColorActive', newValue);
  },

  pointColorHoverHandler: function pointColorHoverHandler(newValue) {
    this.withPropertyChangeHandler('pointColorHover', newValue);
  },

  pointOpacityHandler: function pointOpacityHandler(newValue) {
    this.withPropertyChangeHandler('opacity', newValue);
  },

  pointSizeHandler: function pointSizeHandler(newValue) {
    this.withPropertyChangeHandler('pointSize', newValue);
  },

  pointSizeSelectedHandler: function pointSizeSelectedHandler(newValue) {
    this.withPropertyChangeHandler('pointSizeSelected', newValue);
  },

  pointOutlineWidthHandler: function pointOutlineWidthHandler(newValue) {
    this.withPropertyChangeHandler('pointOutlineWidth', newValue);
  },

  showRecticleHandler: function showRecticleHandler(newValue) {
    this.withPropertyChangeHandler('showRecticle', newValue);
  },

  recticleColorHandler: function recticleColorHandler(newValue) {
    this.withPropertyChangeHandler('recticleColor', newValue);
  },

  cameraTargetHandler: function cameraTargetHandler(newValue) {
    this.withPropertyChangeHandler('cameraTarget', newValue);
  },

  cameraDistanceHandler: function cameraDistanceHandler(newValue) {
    this.withPropertyChangeHandler('cameraDistance', newValue);
  },

  cameraRotationHandler: function cameraRotationHandler(newValue) {
    this.withPropertyChangeHandler('cameraRotation', newValue);
  },

  cameraViewHandler: function cameraViewHandler(newValue) {
    this.withPropertyChangeHandler('cameraView', newValue);
  },

  lassoInitiatorHandler: function lassoInitiatorHandler(newValue) {
    this.withPropertyChangeHandler('lassoInitiator', newValue);
  },

  mouseModeHandler: function mouseModeHandler(newValue) {
    this.withPropertyChangeHandler('mouseMode', newValue);
  },

  otherOptionsHandler: function otherOptionsHandler(newOptions) {
    this.scatterplot.draw(newOptions);
  },

  viewResetHandler: function viewResetHandler() {
    this.scatterplot.reset();
    setTimeout(() => {
      this.model.set('view_reset', false);
      this.model.save_changes();
    }, 0);
  },

  resizeHandler: function resizeHandler() {
    this.width = Math.max(MIN_WIDTH, this.el.getBoundingClientRect().width);
    this.container.style.height = this.height + 'px';
    this.scatterplot.set({
      width: this.width,
      height: this.height
    });
  },

  withPropertyChangeHandler: function withPropertyChangeHandler(property, changedValue) {
    var properties = {};
    properties[property] = changedValue;
    this.scatterplot.set(properties);
  },

  withModelChangeHandler: function withModelChangeHandler(property, handler) {
    var self = this;

    return function modelChangeHandler() {
      var changes = self.model.changedAttributes();
      var pyPropertyName = camelToSnake(property);

      if (
        changes[pyPropertyName] === undefined ||
        self[property + 'Changed'] === true
      ) {
        self[property + 'Changed'] = false;
        return;
      };

      self[property] = changes[camelToSnake(property)];

      if (handler) handler(self[property]);
    }
  }
});

module.exports = {
  JScatterModel: JScatterModel,
  JScatterView: JScatterView
};
