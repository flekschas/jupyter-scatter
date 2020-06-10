var widgets = require('@jupyter-widgets/base');
var _ = require('lodash');
var reglScatterplot = require('regl-scatterplot/dist/regl-scatterplot.js');
var packageJson = require('../package.json');

var createScatterplot = reglScatterplot.default;

var ScatterplotModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(
    _.result(this, 'widgets.DOMWidgetModel.prototype.defaults'),
    {
      _model_name : 'ScatterplotModel',
      _model_module : 'jupyter-scatterplot',
      _model_module_version : packageJson.version,
      _view_name : 'ScatterplotView',
      _view_module : 'jupyter-scatterplot',
      _view_module_version : packageJson.version
    }
  )
});

function camelToSnake(string) {
  return string.replace(/[\w]([A-Z])/g, function(m) {
    return m[0] + "_" + m[1];
  }).toLowerCase();
}

var MIN_WIDTH = 240;

var properties = [
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
  'options',
];

// Custom View. Renders the widget model.
var ScatterplotView = widgets.DOMWidgetView.extend({
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
    this.container.style.border = this.options.theme === 'dark'
      ? '#333333' : '#dddddd';
    this.container.style.borderRadius = '2px';
    this.container.style.height = this.height + 'px';

    this.el.appendChild(this.container);

    this.canvas = document.createElement('canvas');
    this.canvas.style.position = 'absolute';
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';

    this.container.appendChild(this.canvas);

    window.requestAnimationFrame(() => {
      this.width = Math.max(MIN_WIDTH, this.el.getBoundingClientRect().width);

      var initialOptions = {
        canvas: this.canvas,
        width: this.width,
      }

      properties.forEach(function(propertyName) {
        initialOptions[propertyName] = self[propertyName];
      });

      console.log('initialOptions', initialOptions);

      this.scatterplot = createScatterplot(initialOptions);

      console.log(
        'jupyter-scatterplot v' + packageJson.version +
        ' with regl-scatterplot v' + this.scatterplot.get('version')
      );

      this.container.api = this.scatterplot;

      // Listen to events from the JavaScript world
      this.pointoverHandlerBound = this.pointoverHandler.bind(this);
      this.pointoutHandlerBound = this.pointoutHandler.bind(this);
      this.selectHandlerBound = this.selectHandler.bind(this);
      this.deselectHandlerBound = this.deselectHandler.bind(this);
      this.scatterplot.subscribe('pointover', this.pointoverHandlerBound);
      this.scatterplot.subscribe('pointout', this.pointoutHandlerBound);
      this.scatterplot.subscribe('select', this.selectHandlerBound);
      this.scatterplot.subscribe('deselect', this.deselectHandlerBound);

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

      window.addEventListener('resize', this.resizeHandler.bind(this));
      window.addEventListener('deviceorientation', this.resizeHandler.bind(this));

      this.resizeHandler();
      this.colorCanvas();

      if (this.points.length) {
        this.scatterplot
          .draw(this.points)
          .then(() => {
            if (this.selectedPoints.length) {
              this.scatterplot.select(this.selectedPoints);
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
        this.backgroundColor.slice(0, 3).map((x) => x * 255).join(',') +
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

  pointoutHandler: function pointoutHandler(event) {
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

  deselectHandler: function deselectHandler(event) {
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
    this.withPropertyChangeHandler('pointOpacity', newValue);
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

  optionsHandler: function optionsHandler(newOptions) {
    this.scatterplot.draw(newOptions);
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
    var changes = this.model.changedAttributes();

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
  ScatterplotModel: ScatterplotModel,
  ScatterplotView: ScatterplotView
};
