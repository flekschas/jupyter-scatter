const widgets = require('@jupyter-widgets/base');
const reglScatterplot = require('regl-scatterplot/dist/regl-scatterplot.js');
const pubSub = require('pub-sub-es');
const d3Axis = require('d3-axis');
const d3Scale = require('d3-scale');
const d3Selection = require('d3-selection');
const codecs = require('./codecs');
const createLegend = require('./legend');
const packageJson = require('../package.json');

const createScatterplot = reglScatterplot.default;
const createRenderer = reglScatterplot.createRenderer;

const JupyterScatterModel = widgets.DOMWidgetModel.extend(
  {
    defaults: Object.assign(
      {},
      widgets.DOMWidgetModel.prototype.defaults(),
      {
        _model_name : 'JupyterScatterModel',
        _model_module : packageJson.name,
        _model_module_version : packageJson.version,
        _view_name : 'JupyterScatterView',
        _view_module : packageJson.name,
        _view_module_version : packageJson.version,
      }
    )
  },
  {
    serializers: Object.assign(
      {},
      widgets.DOMWidgetModel.serializers,
      {
        points: new codecs.Numpy2D('float32'),
        selection: new codecs.Numpy1D('uint32'),
        view_data: new codecs.Numpy1D('uint8'),
      }
    )
  }
);

const AXES_LABEL_SIZE = 16;
const AXES_PADDING_X = 40;
const AXES_PADDING_X_WITH_LABEL = AXES_PADDING_X + AXES_LABEL_SIZE;
const AXES_PADDING_Y = 20;
const AXES_PADDING_Y_WITH_LABEL = AXES_PADDING_Y + AXES_LABEL_SIZE;

function camelToSnake(string) {
  return string.replace(/[\w]([A-Z])/g, function(m) {
    return m[0] + "_" + m[1];
  }).toLowerCase();
}

function downloadBlob(blob, name) {
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

function getScale(scaleType) {
  if (scaleType.startsWith('log')) {
    return d3Scale.scaleLog().base(scaleType.split('_')[1] || 10);
  }

  if (scaleType.startsWith('pow')) {
    return d3Scale.scalePow().exponent(scaleType.split('_')[1] || 2);
  }

  return d3Scale.scaleLinear();
}

/**
 * This dictionary maps between the camelCased Python property names and their
 * JavaScript counter parts. In most cases the name is identical but they can be
 * different. E.g., size (Python) vs pointSize (JavaScript)
 */
const properties = {
  backgroundColor: 'backgroundColor',
  backgroundImage: 'backgroundImage',
  cameraDistance: 'cameraDistance',
  cameraRotation: 'cameraRotation',
  cameraTarget: 'cameraTarget',
  cameraView: 'cameraView',
  color: 'pointColor',
  colorSelected: 'pointColorActive',
  colorBy: 'colorBy',
  colorHover: 'pointColorHover',
  width: 'width',
  height: 'height',
  lassoColor: 'lassoColor',
  lassoInitiator: 'lassoInitiator',
  lassoMinDelay: 'lassoMinDelay',
  lassoMinDist: 'lassoMinDist',
  mouseMode: 'mouseMode',
  opacity: 'opacity',
  opacityBy: 'opacityBy',
  opacityUnselected: 'opacityInactiveScale',
  otherOptions: 'otherOptions',
  points: 'points',
  reticle: 'showReticle',
  reticleColor: 'reticleColor',
  selection: 'selectedPoints',
  size: 'pointSize',
  sizeBy: 'sizeBy',
  connect: 'showPointConnections',
  connectionColor: 'pointConnectionColor',
  connectionColorSelected: 'pointConnectionColorActive',
  connectionColorHover: 'pointConnectionColorHover',
  connectionColorBy: 'pointConnectionColorBy',
  connectionOpacity: 'pointConnectionOpacity',
  connectionOpacityBy: 'pointConnectionOpacityBy',
  connectionSize: 'pointConnectionSize',
  connectionSizeBy: 'pointConnectionSizeBy',
  viewDownload: 'viewDownload',
  viewReset: 'viewReset',
  hovering: 'hovering',
  axes: 'axes',
  axesColor: 'axesColor',
  axesGrid: 'axesGrid',
  axesLabels: 'axesLabels',
  legend: 'legend',
  legendSize: 'legendSize',
  legendColor: 'legendColor',
  legendPosition: 'legendPosition',
  legendEncoding: 'legendEncoding',
  xScale: 'xScale',
  yScale: 'yScale',
};

const reglScatterplotProperty = new Set([
  'backgroundColor',
  'backgroundImage',
  'cameraDistance',
  'cameraRotation',
  'cameraTarget',
  'cameraView',
  'pointColor',
  'pointColorActive',
  'colorBy',
  'pointColorHover',
  'width',
  'height',
  'lassoColor',
  'lassoInitiator',
  'lassoMinDelay',
  'lassoMinDist',
  'mouseMode',
  'opacity',
  'opacityBy',
  'opacityInactiveScale',
  'points',
  'showReticle',
  'reticleColor',
  'selectedPoints',
  'pointSize',
  'sizeBy',
  'showPointConnections',
  'pointConnectionColor',
  'pointConnectionColorActive',
  'pointConnectionColorHover',
  'pointConnectionColorBy',
  'pointConnectionOpacity',
  'pointConnectionOpacityBy',
  'pointConnectionSize',
  'pointConnectionSizeBy',
]);

// Custom View. Renders the widget model.
const JupyterScatterView = widgets.DOMWidgetView.extend({
  render: function render() {
    var self = this;

    if (!window.jupyterScatter) {
      window.jupyterScatter = {
        renderer: createRenderer(),
        versionLog: false,
      }
    }

    Object.keys(properties).forEach(function(propertyName) {
      self[propertyName] = self.model.get(camelToSnake(propertyName));
    });

    this.width = !Number.isNaN(+this.model.get('width')) && +this.model.get('width') > 0
      ? +this.model.get('width')
      : 'auto';

    // Create a random 6-letter string
    // From https://gist.github.com/6174/6062387
    this.randomStr = (
      Math.random().toString(36).substring(2, 5) +
      Math.random().toString(36).substring(2, 5)
    );
    this.model.set('dom_element_id', this.randomStr);

    this.container = document.createElement('div');
    this.container.setAttribute('id', this.randomStr);
    this.container.style.position = 'relative'
    this.container.style.width = this.width === 'auto'
      ? '100%'
      : this.width + 'px';
    this.container.style.height = this.model.get('height') + 'px';
    this.el.appendChild(this.container);

    this.canvasWrapper = document.createElement('div');
    this.canvasWrapper.style.position = 'absolute';
    this.canvasWrapper.style.top = '0';
    this.canvasWrapper.style.left = '0';
    this.canvasWrapper.style.right = '0';
    this.canvasWrapper.style.bottom = '0';
    this.container.appendChild(this.canvasWrapper);

    this.canvas = document.createElement('canvas');
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.canvasWrapper.appendChild(this.canvas);

    window.requestAnimationFrame(function init() {
      const initialOptions = {
        renderer: window.jupyterScatter.renderer,
        canvas: self.canvas,
      }

      if (self.width !== 'auto') initialOptions.width = self.width;

      Object.entries(properties).forEach(function(property) {
        const pyName = property[0];
        const jsName = property[1];
        if (self[pyName] !== null && reglScatterplotProperty.has(jsName))
          initialOptions[jsName] = self[pyName];
      });

      self.scatterplot = createScatterplot(initialOptions);

      if (!window.jupyterScatter.versionLog) {
        // eslint-disable-next-line
        console.log(
          'jupyter-scatter v' + packageJson.version +
          ' with regl-scatterplot v' + self.scatterplot.get('version')
        );
        window.jupyterScatter.versionLog = true;
      }

      self.container.api = self.scatterplot;

      if (self.model.get('axes')) self.createAxes();
      if (self.model.get('axes_grid')) self.createAxesGrid();
      if (self.model.get('legend')) self.showLegend();

      // Listen to events from the JavaScript world
      self.pointoverHandlerBound = self.pointoverHandler.bind(self);
      self.pointoutHandlerBound = self.pointoutHandler.bind(self);
      self.selectHandlerBound = self.selectHandler.bind(self);
      self.deselectHandlerBound = self.deselectHandler.bind(self);
      self.externalViewChangeHandlerBound = self.externalViewChangeHandler.bind(self);
      self.viewChangeHandlerBound = self.viewChangeHandler.bind(self);
      self.resizeHandlerBound = self.resizeHandler.bind(self);

      self.scatterplot.subscribe('pointover', self.pointoverHandlerBound);
      self.scatterplot.subscribe('pointout', self.pointoutHandlerBound);
      self.scatterplot.subscribe('select', self.selectHandlerBound);
      self.scatterplot.subscribe('deselect', self.deselectHandlerBound);
      self.scatterplot.subscribe('view', self.viewChangeHandlerBound);

      pubSub.globalPubSub.subscribe(
        'jscatter::view', self.externalViewChangeHandlerBound
      );

      if ('ResizeObserver' in window) {
        self.canvasObserver = new ResizeObserver(self.resizeHandlerBound);
        self.canvasObserver.observe(self.canvas);
      } else {
        window.addEventListener('resize', self.resizeHandlerBound);
        window.addEventListener('orientationchange', self.resizeHandlerBound);
      }

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

      self.colorCanvas();

      if (self.points.length) {
        self.scatterplot
          .draw(self.points)
          .then(function onInitialDraw() {
            if (self.selection.length) {
              self.scatterplot.select(self.selection, { preventEvent: true });
            }
          });
      }
    });

    this.model.save_changes();
  },

  getOuterDimensions: function getOuterDimensions() {
    let xPadding = 0;
    let yPadding = 0;

    if (this.model.get('axes')) {
      const labels = this.model.get('axes_labels');
      xPadding = labels ? AXES_PADDING_X_WITH_LABEL : AXES_PADDING_X;
      yPadding = labels ? AXES_PADDING_Y_WITH_LABEL : AXES_PADDING_Y;
    }

    const outerWidth = this.model.get('width') === 'auto'
      ? this.container.getBoundingClientRect().width
      : this.model.get('width') + xPadding;

    const outerHeight = this.model.get('height') + yPadding;

    return [outerWidth, outerHeight]
  },

  createAxes: function createAxes() {
    this.axesSvg = d3Selection.select(this.container).select('svg').node()
      ? d3Selection.select(this.container).select('svg')
      : d3Selection.select(this.container).append('svg');
    this.axesSvg.style('top', 0);
    this.axesSvg.style('left', 0);
    this.axesSvg.style('width', '100%');
    this.axesSvg.style('height', '100%');
    this.axesSvg.style('pointer-events', 'none');
    this.axesSvg.style('user-select', 'none');
    const color = this.model.get('axes_color')
      .map(function (c) { return Math.round(c * 255); });
    this.axesSvg.style('color', `rgba(${color[0]}, ${color[1]}, ${color[2]}, 1)`);

    const [width, height] = this.getOuterDimensions();

    const currentXScaleRegl = this.scatterplot.get('xScale');
    const currentYScaleRegl = this.scatterplot.get('yScale');

    const labels = this.model.get('axes_labels');
    const xPadding = labels ? AXES_PADDING_X_WITH_LABEL : AXES_PADDING_X;
    const yPadding = labels ? AXES_PADDING_Y_WITH_LABEL : AXES_PADDING_Y;

    // Regl-Scatterplot's gl-space is always linear, hence we have to pass a
    // linear scale to regl-scatterplot.
    // In the future we might integrate this into regl-scatterplot directly
    this.xScaleRegl = d3Scale.scaleLinear()
      .domain(this.model.get('x_domain'))
      .range([0, width - xPadding]);
    // This scale is used for the D3 axis
    this.xScaleAxis = getScale(this.model.get('x_scale'))
      .domain(this.model.get('x_domain'))
      .range([0, width - xPadding]);
    // This scale converts between the linear, log, or power normalized data
    // scale and the axis
    this.xScaleRegl2Axis = getScale(this.model.get('x_scale'))
      .domain(this.model.get('x_domain'))
      .range(this.model.get('x_domain'));

    this.yScaleRegl = d3Scale.scaleLinear()
      .domain(this.model.get('y_domain'))
      .range([height - yPadding, 0]);
    this.yScaleAxis = getScale(this.model.get('y_scale'))
      .domain(this.model.get('y_domain'))
      .range([height - yPadding, 0]);
    this.yScaleRegl2Axis = getScale(this.model.get('y_scale'))
      .domain(this.model.get('y_domain'))
      .range(this.model.get('y_domain'));

    if (currentXScaleRegl) {
      this.xScaleAxis.domain(
        currentXScaleRegl.domain().map(this.xScaleRegl2Axis.invert)
      );
    }

    if (currentYScaleRegl) {
      this.yScaleAxis.domain(
        currentYScaleRegl.domain().map(this.yScaleRegl2Axis.invert)
      );
    }

    this.xAxis = d3Axis.axisBottom(this.xScaleAxis);
    this.yAxis = d3Axis.axisRight(this.yScaleAxis);

    this.xAxisContainer = this.axesSvg.select('.x-axis').node()
      ? this.axesSvg.select('.x-axis')
      : this.axesSvg.append('g').attr('class', 'x-axis');

    this.xAxisContainer
      .attr('transform', `translate(0, ${height - yPadding})`)
      .call(this.xAxis);

    this.yAxisContainer = this.axesSvg.select('.y-axis').node()
      ? this.axesSvg.select('.y-axis')
      : this.axesSvg.append('g').attr('class', 'y-axis');

    this.yAxisContainer
      .attr('transform', `translate(${width - xPadding}, 0)`)
      .call(this.yAxis);

    this.axesSvg.selectAll('.domain').attr('opacity', 0);

    if (labels) {
      this.xAxisLabel = this.axesSvg.select('.x-axis-label').node()
        ? this.axesSvg.select('.x-axis-label')
        : this.axesSvg.append('text').attr('class', 'x-axis-label');

      this.xAxisLabel
        .text(labels[0])
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('x', (width - xPadding) / 2)
        .attr('y', height);

      this.yAxisLabel = this.axesSvg.select('.y-axis-label').node()
        ? this.axesSvg.select('.y-axis-label')
        : this.axesSvg.append('text').attr('class', 'y-axis-label');

      this.yAxisLabel
        .text(labels[1])
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'hanging')
        .attr('x', (height - yPadding) / 2)
        .attr('y', -width)
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('transform', `rotate(90)`);
    }

    this.updateContainerDimensions();

    this.scatterplot.set({
      xScale: this.xScaleRegl,
      yScale: this.yScaleRegl,
    });

    this.canvasWrapper.style.right = `${xPadding}px`;
    this.canvasWrapper.style.bottom = `${yPadding}px`;

    if (this.model.get('axes_grid')) this.createAxesGrid();

    this.updateLegendWrapperPosition();
  },

  removeAxes: function removeAxes() {
    this.axesSvg.node().remove();
    this.axesSvg = undefined;
    this.xAxis = undefined;
    this.yAxis = undefined;
    this.xAxisContainer = undefined;
    this.yAxisContainer = undefined;
    this.xAxisContainer = undefined;
    this.xAxisLabel = undefined;
    this.yAxisLabel = undefined;

    this.canvasWrapper.style.top = '0';
    this.canvasWrapper.style.left = '0';
    this.canvasWrapper.style.right = '0';
    this.canvasWrapper.style.bottom = '0';

    this.updateContainerDimensions();

    this.scatterplot.set({
      xScale: undefined,
      yScale: undefined,
    });
  },

  createAxesGrid: function createAxesGrid() {
    const { width, height } = this.canvasWrapper.getBoundingClientRect();
    if (this.xAxis) {
      this.xAxis.tickSizeInner(-height);
      this.xAxisContainer.call(this.xAxis);
    }
    if (this.yAxis) {
      this.yAxis.tickSizeInner(-width);
      this.yAxisContainer.call(this.yAxis);
    }
    if (this.axesSvg) {
      this.axesSvg.selectAll('line')
        .attr('stroke-opacity', 0.2)
        .attr('stroke-dasharray', 2);
    }
  },

  removeAxesGrid: function removeAxesGrid() {
    if (this.xAxis) {
      this.xAxis.tickSizeInner(6);
      this.xAxisContainer.call(this.xAxis);
    }
    if (this.yAxis) {
      this.yAxis.tickSizeInner(6);
      this.yAxisContainer.call(this.yAxis);
    }
    if (this.axesSvg) {
      this.axesSvg.selectAll('line')
        .attr('stroke-opacity', null)
        .attr('stroke-dasharray', null);
    }
  },

  showLegend: function showLegend() {
    this.hideLegend();

    this.legendWrapper = document.createElement('div');
    this.legendWrapper.className = 'legend-wrapper';
    this.legendWrapper.style.position = 'absolute';
    this.legendWrapper.style.pointerEvents = 'none';
    this.updateLegendWrapperPosition();

    this.legend = createLegend(
      this.model.get('legend_encoding'),
      this.model.get('legend_color'),
      this.model.get('background_color'),
      this.model.get('legend_size')
    );
    this.updateLegendPosition();

    this.legendWrapper.appendChild(this.legend);
    this.container.appendChild(this.legendWrapper);
  },

  hideLegend: function hideLegend() {
    if (!this.legendWrapper) return;
    this.container.removeChild(this.legendWrapper);
    this.legendWrapper = undefined;
    this.legend = undefined;
  },

  updateLegendWrapperPosition: function updateLegendWrapperPosition() {
    if (!this.legendWrapper) return;

    const labels = this.model.get('axes_labels');
    const xPadding = labels ? AXES_PADDING_X_WITH_LABEL : AXES_PADDING_X;
    const yPadding = labels ? AXES_PADDING_Y_WITH_LABEL : AXES_PADDING_Y;

    this.legendWrapper.style.top = 0;
    this.legendWrapper.style.bottom = yPadding + 'px';
    this.legendWrapper.style.left = 0;
    this.legendWrapper.style.right = xPadding + 'px';
  },

  updateLegendPosition: function updateLegendPosition() {
    if (!this.legend) return;

    this.legend.style.position = 'absolute';
    this.legend.style.top = null;
    this.legend.style.bottom = null;
    this.legend.style.left = null;
    this.legend.style.right = null;
    this.legend.style.transform = null;

    const position = this.model.get('legend_position');
    let translateX = 0;
    let translateY = 0;

    if (position.indexOf('top') >= 0) {
      this.legend.style.top = 0;
    } else if (position.indexOf('bottom') >= 0) {
      this.legend.style.bottom = 0;
    } else {
      this.legend.style.top = '50%';
      translateY = '-50%';
    }

    if (position.indexOf('left') >= 0) {
      this.legend.style.left = 0;
    } else if (position.indexOf('right') >= 0) {
      this.legend.style.right = 0;
    } else {
      this.legend.style.left = '50%';
      translateX = '-50%';
    }

    if (translateX || translateY) {
      this.legend.style.transform = `translate(${translateX}, ${translateY})`;
    }
  },

  updateContainerDimensions: function updateContainerDimensions() {
    const width = this.model.get('width');
    const height = this.model.get('height');

    let xPadding = 0;
    let yPadding = 0;

    if (this.model.get('axes')) {
      const labels = this.model.get('axes_labels');
      xPadding = labels ? AXES_PADDING_X_WITH_LABEL : AXES_PADDING_X;
      yPadding = labels ? AXES_PADDING_Y_WITH_LABEL : AXES_PADDING_Y;
    }

    this.container.style.width = width === 'auto'
      ? '100%'
      : (width + xPadding) + 'px';
    this.container.style.height = (height + yPadding) + 'px';

    window.requestAnimationFrame(() => { this.resizeHandler(); });
  },

  resizeHandler: function resizeHandler() {
    if (!this.model.get('axes')) return;

    const [width, height] = this.getOuterDimensions();

    const labels = this.model.get('axes_labels');
    const xPadding = labels ? AXES_PADDING_X_WITH_LABEL : AXES_PADDING_X;
    const yPadding = labels ? AXES_PADDING_Y_WITH_LABEL : AXES_PADDING_Y;

    this.xScaleAxis.range([0, width - xPadding]);
    this.yScaleAxis.range([height - yPadding, 0]);
    this.xAxis.scale(this.xScaleAxis);
    this.yAxis.scale(this.yScaleAxis);

    this.xAxisContainer
      .attr('transform', `translate(0, ${height - yPadding})`)
      .call(this.xAxis);
    this.yAxisContainer
      .attr('transform', `translate(${width - xPadding}, 0)`)
      .call(this.yAxis);

    this.updateLegendWrapperPosition();

    this.withPropertyChangeHandler('width', this.model.get('width') || 'auto');
    this.withPropertyChangeHandler('height', this.model.get('height'));

    // Render grid
    if (this.model.get('axes_grid')) {
      this.xAxis.tickSizeInner(-(height - yPadding));
      this.yAxis.tickSizeInner(-(width - xPadding));
    }

    if (labels) {
      this.xAxisLabel.attr('x', (width - xPadding) / 2).attr('y', height);
      this.yAxisLabel.attr('x', (height - yPadding) / 2).attr('y', -width);
    }
  },

  remove: function destroy() {
    if (this.canvasObserver) {
      this.canvasObserver.disconnect();
    } else {
      window.removeEventListener('resize', this.resizeHandlerBound);
      window.removeEventListener('orientationchange', this.resizeHandlerBound);
    }
    pubSub.globalPubSub.unsubscribe(
      'jscatter::view',
      this.externalViewChangeHandlerBound
    );
    this.scatterplot.unsubscribe('pointover', this.pointoverHandlerBound);
    this.scatterplot.unsubscribe('pointout', this.pointoutHandlerBound);
    this.scatterplot.unsubscribe('select', this.selectHandlerBound);
    this.scatterplot.unsubscribe('deselect', this.deselectHandlerBound);
    this.scatterplot.unsubscribe('view', this.viewChangeHandlerBound);
    this.scatterplot.destroy();
  },

  // Helper
  colorCanvas: function colorCanvas() {
    if (Array.isArray(this.backgroundColor)) {
      this.container.style.backgroundColor = 'rgb(' +
        this.backgroundColor.slice(0, 3).map(function (x) { return x * 255 }).join(',') +
        ')';
    } else {
      this.container.style.backgroundColor = this.backgroundColor;
    }
  },

  // Event handlers for JS-triggered events
  pointoverHandler: function pointoverHandler(pointIndex) {
    this.hoveringChangedByJs = true;
    this.model.set('hovering', pointIndex);
    this.model.save_changes();
  },

  pointoutHandler: function pointoutHandler() {
    this.hoveringChangedByJs = true;
    this.model.set('hovering', null);
    this.model.save_changes();
  },

  selectHandler: function selectHandler(event) {
    this.selectionChangedByJs = true;
    this.model.set('selection', [...event.points]);
    this.model.save_changes();
  },

  deselectHandler: function deselectHandler() {
    this.selectionChangedByJs = true;
    this.model.set('selection', []);
    this.model.save_changes();
  },

  externalViewChangeHandler: function externalViewChangeHandler(event) {
    const viewSync = this.model.get('view_sync');
    if (
      !viewSync
      || event.uuid !== viewSync
      || event.src === this.randomStr
    ) return;
    this.scatterplot.view(event.view, { preventEvent: true });
  },

  viewChangeHandler: function viewChangeHandler(event) {
    const viewSync = this.model.get('view_sync');
    if (viewSync) {
      pubSub.globalPubSub.publish(
        'jscatter::view',
        {
          src: this.randomStr,
          uuid: viewSync,
          view: event.view,
        }
      );
    }
    if (this.model.get('axes')) {
      this.xScaleAxis.domain(event.xScale.domain().map(this.xScaleRegl2Axis.invert));
      this.yScaleAxis.domain(event.yScale.domain().map(this.yScaleRegl2Axis.invert));

      this.xAxisContainer.call(this.xAxis.scale(this.xScaleAxis));
      this.yAxisContainer.call(this.yAxis.scale(this.yScaleAxis));

      if (this.model.get('axes_grid')) {
        this.axesSvg.selectAll('line')
          .attr('stroke-opacity', 0.2)
          .attr('stroke-dasharray', 2);
      }
    }
  },

  xScaleHandler: function xScaleHandler() {
    this.createAxes();
  },

  yScaleHandler: function yScaleHandler() {
    this.createAxes();
  },

  // Event handlers for Python-triggered events
  pointsHandler: function pointsHandler(newPoints) {
    this.scatterplot.draw(newPoints, {
      transition: true,
      transitionDuration: 3000,
      transitionEasing: 'quadInOut',
    });
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

  hoveringHandler: function hoveringHandler(newHovering) {
    // Avoid calling `this.scatterplot.hover()` twice when the hovering was
    // triggered by the JavaScript (i.e., the user interactively selected points)
    if (this.hoveringChangedByJs) {
      this.hoveringChangedByJs = undefined;
      return;
    }

    if (Number.isNaN(+newHovering)) {
      this.scatterplot.hover({ preventEvent: true });
    } else {
      this.scatterplot.hover(+newHovering, { preventEvent: true });
    }
  },

  widthHandler: function widthHandler() {
    this.updateContainerDimensions();
  },

  heightHandler: function heightHandler() {
    this.updateContainerDimensions();
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

  colorSelectedHandler: function colorSelectedHandler(newValue) {
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

  opacityUnselectedHandler: function opacityUnselectedHandler(newValue) {
    this.withPropertyChangeHandler('opacityInactiveScale', newValue);
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

  connectHandler: function connectHandler(newValue) {
    this.withPropertyChangeHandler('showPointConnections', Boolean(newValue));
  },

  connectionColorHandler: function connectionColorHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionColor', newValue);
  },

  connectionColorSelectedHandler: function connectionColorSelectedHandler(newValue) {
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

  axesHandler: function axesHandler(newValue) {
    if (newValue) this.createAxes();
    else this.removeAxes();
  },

  axesColorHandler: function axesColorHandler() {
    this.createAxes();
  },

  axesGridHandler: function axesGridHandler(newValue) {
    if (newValue) this.createAxesGrid();
    else this.removeAxesGrid();
  },

  axesLabelsHandler: function axesLabelsHandler(newValue) {
    if (!newValue) this.removeAxes();
    this.createAxes();
  },

  legendHandler: function legendHandler(newValue) {
    if (newValue) this.showLegend();
    else this.hideLegend();
  },

  legendColorHandler: function legendColorHandler() {
    this.hideLegend();
    this.showLegend();
  },

  legendSizeHandler: function legendSizeHandler() {
    this.hideLegend();
    this.showLegend();
  },

  legendPositionHandler: function legendPositionHandler() {
    this.updateLegendPosition();
  },

  legendEncodingHandler: function legendEncodingHandler() {
    if (!this.model.get('legend')) return;
    this.showLegend();
  },

  otherOptionsHandler: function otherOptionsHandler(newOptions) {
    this.scatterplot.set(newOptions);
  },

  viewDownloadHandler: function viewDownloadHandler(target) {
    if (!target) return;

    if (target === 'property') {
      const image = this.scatterplot.export();
      this.model.set('view_data', image.data);
      this.model.set('view_shape', [image.width, image.height]);
      this.model.set('view_download', null);
      this.model.save_changes();
      return;
    }

    this.scatterplot.get('canvas').toBlob((blob) => {
      downloadBlob(blob, 'scatter.png');
      setTimeout(() => {
        this.model.set('view_download', null);
        this.model.save_changes();
      }, 0);
    });
  },

  viewResetHandler: function viewResetHandler() {
    this.scatterplot.reset();
    setTimeout(() => {
      this.model.set('view_reset', false);
      this.model.save_changes();
    }, 0);
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
