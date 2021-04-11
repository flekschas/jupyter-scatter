/* eslint-env browser */
const widgets = require('@jupyter-widgets/base');
const _ = require('lodash');
const reglScatterplot = require('regl-scatterplot/dist/regl-scatterplot.js');
const packageJson = require('../package.json');

const createScatterplot = reglScatterplot.default;

const JupyterScatterModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(
    _.result(this, 'widgets.DOMWidgetModel.prototype.defaults'),
    {
      _model_name : 'JupyterScatterModel',
      _model_module : packageJson.name,
      _model_module_version : packageJson.version,
      _view_name : 'JupyterScatterView',
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

function flipObj(obj) {
  return Object.entries(obj).reduce((ret, entry) => {
    const [ key, value ] = entry;
    ret[ value ] = key;
    return ret;
  }, {});
}

const MIN_WIDTH = 240;

const properties = {
  backgroundColor: 'backgroundColor',
  backgroundImage: 'backgroundImage',
  cameraDistance: 'cameraDistance',
  cameraRotation: 'cameraRotation',
  cameraTarget: 'cameraTarget',
  cameraView: 'cameraView',
  color: 'pointColor',
  colorActive: 'pointColorActive',
  colorBy: 'colorBy',
  colorHover: 'pointColorHover',
  height: 'height',
  lassoColor: 'lassoColor',
  lassoInitiator: 'lassoInitiator',
  lassoMinDelay: 'lassoMinDelay',
  lassoMinDist: 'lassoMinDist',
  mouseMode: 'mouseMode',
  opacity: 'opacity',
  opacityBy: 'opacityBy',
  otherOptions: 'otherOptions',
  points: 'points',
  reticle: 'showReticle',
  reticleColor: 'reticleColor',
  selection: 'selectedPoints',
  selectionOutlineWidth: 'pointOutlineWidth',
  selectionSizeAddition: 'pointSizeSelected',
  size: 'pointSize',
  sizeBy: 'sizeBy',
  connect: 'showPointConnections',
  connectionColor: 'pointConnectionColor',
  connectionColorActive: 'pointConnectionColorActive',
  connectionColorHover: 'pointConnectionColorHover',
  connectionColorBy: 'pointConnectionColorBy',
  connectionOpacity: 'pointConnectionOpacity',
  connectionOpacityBy: 'pointConnectionOpacityBy',
  connectionSize: 'pointConnectionSize',
  connectionSizeBy: 'pointConnectionSizeBy',
  viewReset: 'viewReset',
  sortOrder: 'sortOrder'
};

// Custom View. Renders the widget model.
const JupyterScatterView = widgets.DOMWidgetView.extend({
  render: function render() {
    var self = this;

    Object.keys(properties).forEach(function(propertyName) {
      self[propertyName] = self.model.get(camelToSnake(propertyName));
    });

    this.height = this.model.get('height');

    if (self.sortOrder) this.inverseSortOrder = flipObj(self.sortOrder);

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

      Object.entries(properties).forEach(function(property) {
        const pyName = property[0];
        const jsName = property[1];
        if (self[pyName] !== null)
          initialOptions[jsName] = self[pyName];
      });

      self.scatterplot = createScatterplot(initialOptions);

      // eslint-disable-next-line
      console.log(
        'jupyterscatter v' + packageJson.version +
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
      Object.keys(properties).forEach(function(propertyName) {
        if (self[propertyName + 'Handler']) {
          self.model.on(
            'change:' + camelToSnake(propertyName),
            self.withModelChangeHandler(
              propertyName,
              self[propertyName + 'Handler'].bind(self)
            ),
            self
          );
        } else {
          console.warn('No handler for ' + propertyName);
        }
      });

      window.addEventListener('resize', self.resizeHandler.bind(self));
      window.addEventListener('deviceorientation', self.resizeHandler.bind(self));

      self.resizeHandler();
      self.colorCanvas();

      if (self.points.length) {
        self.scatterplot
          .draw(self.points)
          .then(function onInitialDraw() {
            if (self.selection.length) {
              self.scatterplot.select(self.sortedSelection());
            }
          });
      }
    });

    this.model.save_changes();
  },

  remove: function destroy() {
    this.scatterplot.destroy();
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

  sortSelection: function sortSelection() {
    if (this.sortOrder) {
      return self.selection.map((s) => this.sortOrder[s]);
    }
    return self.selection;
  },

  // Event handlers for JS-triggered events
  pointoverHandler: function pointoverHandler(pointIndex) {
    this.model.set('hovering', pointIndex);
    this.model.save_changes();
  },

  pointoutHandler: function pointoutHandler() {
    this.model.set('hovering', null);
    this.model.save_changes();
  },

  selectHandler: function selectHandler(event) {
    this.selectionChangedByJs = true;
    if (this.inverseSortOrder) {
      this.model.set('selection', event.points.map(point => this.inverseSortOrder[point]));
    } else {
      this.model.set('selection', [...event.points]);
    }

    this.model.save_changes();
  },

  deselectHandler: function deselectHandler() {
    this.selectionChangedByJs = true;
    this.model.set('selection', []);
    this.model.save_changes();
  },

  // Event handlers for Python-triggered events
  pointsHandler: function pointsHandler(newPoints) {
    this.scatterplot.draw(newPoints);
  },

  selectionHandler: function selectionHandler(newSelection) {
    // Avoid calling `this.scatterplot.select()` twice when the selection was
    // triggered by the JavaScript (i.e., the user interactively selected points)
    if (this.selectionChangedByJs) {
      this.selectionChangedByJs = undefined;
      return;
    }

    if (!newSelection || !newSelection.length) {
      this.scatterplot.deselect({ preventEvent: true });
    } else {
      this.scatterplot.select(newSelection, { preventEvent: true });
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

  lassoColorHandler: function lassoColorHandler(newValue) {
    this.withPropertyChangeHandler('lassoColor', newValue);
  },

  lassoMinDelayHandler: function lassoMinDelayHandler(newValue) {
    this.withPropertyChangeHandler('lassoMinDelay', newValue);
  },

  lassoMinDistHandler: function lassoMinDistHandler(newValue) {
    this.withPropertyChangeHandler('lassoMinDist', newValue);
  },

  colorHandler: function colorHandler(newValue) {
    this.withPropertyChangeHandler('pointColor', newValue);
  },

  colorActiveHandler: function colorActiveHandler(newValue) {
    this.withPropertyChangeHandler('pointColorActive', newValue);
  },

  colorHoverHandler: function colorHoverHandler(newValue) {
    this.withPropertyChangeHandler('pointColorHover', newValue);
  },

  colorByHandler: function colorByHandler(newValue) {
    this.withPropertyChangeHandler('colorBy', newValue);
  },

  opacityHandler: function opacityHandler(newValue) {
    this.withPropertyChangeHandler('opacity', newValue);
  },

  opacityByHandler: function opacityByHandler(newValue) {
    this.withPropertyChangeHandler('opacityBy', newValue);
  },

  sizeHandler: function sizeHandler(newValue) {
    this.withPropertyChangeHandler('pointSize', newValue);
  },

  sizeByHandler: function sizeByHandler(newValue) {
    this.withPropertyChangeHandler('sizeBy', newValue);
  },

  selectionSizeAdditionHandler: function selectionSizeAdditionHandler(newValue) {
    this.withPropertyChangeHandler('pointSizeSelected', newValue);
  },

  selectionOutlineWidthHandler: function selectionOutlineWidthHandler(newValue) {
    this.withPropertyChangeHandler('pointOutlineWidth', newValue);
  },

  connectHandler: function connectHandler(newValue) {
    this.withPropertyChangeHandler('showPointConnections', Boolean(newValue));
  },

  connectionColorHandler: function connectionColorHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionColor', newValue);
  },

  connectionColorActiveHandler: function connectionColorActiveHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionColorActive', newValue);
  },

  connectionColorHoverHandler: function connectionColorHoverHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionColorHover', newValue);
  },

  connectionColorByHandler: function connectionColorByHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionColorBy', newValue);
  },

  connectionOpacityHandler: function connectionOpacityHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionOpacity', newValue);
  },

  connectionOpacityByHandler: function connectionOpacityByHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionOpacityBy', newValue);
  },

  connectionSizeHandler: function connectionSizeHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionSize', newValue);
  },

  connectionSizeByHandler: function connectionSizeByHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionSizeBy', newValue);
  },

  reticleHandler: function reticleHandler(newValue) {
    this.withPropertyChangeHandler('showReticle', newValue);
  },

  reticleColorHandler: function reticleColorHandler(newValue) {
    this.withPropertyChangeHandler('reticleColor', newValue);
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
    this.scatterplot.set(newOptions);
  },

  sortOrderHandler: function sortOrderHandler(newSortOrder) {
    this.sortOrder = newSortOrder;
    if (self.sortOrder) {
      this.inverseSortOrder = flipObj(self.sortOrder);
    } else {
      this.inverseSortOrder = undefined;
    }
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
    var p = {};
    p[property] = changedValue;
    this.scatterplot.set(p);
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
  JupyterScatterModel: JupyterScatterModel,
  JupyterScatterView: JupyterScatterView
};
