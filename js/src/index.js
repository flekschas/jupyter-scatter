import * as reglScatterplot from 'regl-scatterplot';
import * as pubSub from 'pub-sub-es';

import * as d3Axis from 'd3-axis';
import * as d3Scale from 'd3-scale';
import * as d3Selection from 'd3-selection';

import * as codecs from "./codecs.js";
import { createLegend } from "./legend.js";
import { version } from "../package.json";

const createScatterplot = reglScatterplot.default;
const createRenderer = reglScatterplot.createRenderer;

const AXES_LABEL_SIZE = 16;
const AXES_PADDING_X = 40;
const AXES_PADDING_X_WITH_LABEL = AXES_PADDING_X + AXES_LABEL_SIZE;
const AXES_PADDING_Y = 20;
const AXES_PADDING_Y_WITH_LABEL = AXES_PADDING_Y + AXES_LABEL_SIZE;

function camelToSnake(string) {
  return string.replace(/[\w]([A-Z])/g, (m) => m[0] + "_" + m[1]).toLowerCase();
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
  lassoOnLongPress: 'lassoOnLongPress',
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
  filter: 'filteredPoints',
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
  viewSync: 'viewSync',
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
  zoomTo: 'zoomTo',
  zoomToCallIdx: 'zoomToCallIdx',
  zoomAnimation: 'zoomAnimation',
  zoomPadding: 'zoomPadding',
  zoomOnSelection: 'zoomOnSelection',
  zoomOnFilter: 'zoomOnFilter',
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
  'lassoOnLongPress',
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
  'filteredPoints',
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
class JupyterScatterView {

  constructor({ el, model }) {
    this.el = el;
    this.model = model;
  }

  render() {
    if (!window.jupyterScatter) {
      window.jupyterScatter = {
        renderer: createRenderer(),
        versionLog: false,
      }
    }

    Object.keys(properties).forEach((propertyName) => {
      this[propertyName] = this.model.get(camelToSnake(propertyName));
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

    window.requestAnimationFrame(() => {
      const initialOptions = {
        renderer: window.jupyterScatter.renderer,
        canvas: this.canvas,
        keyMap: { shift: 'merge' },
      }

      if (this.width !== 'auto') initialOptions.width = this.width;

      Object.entries(properties).forEach((property) => {
        const pyName = property[0];
        const jsName = property[1];
        if (this[pyName] !== null && reglScatterplotProperty.has(jsName))
          initialOptions[jsName] = this[pyName];
      });

      this.scatterplot = createScatterplot(initialOptions);

      if (!window.jupyterScatter.versionLog) {
        // eslint-disable-next-line
        console.log(
          'jupyter-scatter v' + version +
          ' with regl-scatterplot v' + this.scatterplot.get('version')
        );
        window.jupyterScatter.versionLog = true;
      }

      this.container.api = this.scatterplot;

      if (this.model.get('axes')) this.createAxes();
      if (this.model.get('axes_grid')) this.createAxesGrid();
      if (this.model.get('legend')) this.showLegend();

      // Listen to events from the JavaScript world
      this.pointoverHandlerBound = this.pointoverHandler.bind(this);
      this.pointoutHandlerBound = this.pointoutHandler.bind(this);
      this.selectHandlerBound = this.selectHandler.bind(this);
      this.deselectHandlerBound = this.deselectHandler.bind(this);
      this.filterEventHandlerBound = this.filterEventHandler.bind(this);
      this.externalViewChangeHandlerBound = this.externalViewChangeHandler.bind(this);
      this.viewChangeHandlerBound = this.viewChangeHandler.bind(this);
      this.resizeHandlerBound = this.resizeHandler.bind(this);

      this.scatterplot.subscribe('pointover', this.pointoverHandlerBound);
      this.scatterplot.subscribe('pointout', this.pointoutHandlerBound);
      this.scatterplot.subscribe('select', this.selectHandlerBound);
      this.scatterplot.subscribe('deselect', this.deselectHandlerBound);
      this.scatterplot.subscribe('filter', this.filterEventHandlerBound);
      this.scatterplot.subscribe('view', this.viewChangeHandlerBound);

      window.pubSub = pubSub;

      this.viewSyncHandler(this.model.get('view_sync'));

      if ('ResizeObserver' in window) {
        this.canvasObserver = new ResizeObserver(this.resizeHandlerBound);
        this.canvasObserver.observe(this.canvas);
      } else {
        window.addEventListener('resize', this.resizeHandlerBound);
        window.addEventListener('orientationchange', this.resizeHandlerBound);
      }

      // Listen to messages from the Python world
      Object.keys(properties).forEach((propertyName) => {
        const handler = this[propertyName + 'Handler'];
        if (!handler) return;
        this.model.on(`change:${camelToSnake(propertyName)}`, () => {
          this[propertyName] = this.model.get(camelToSnake(propertyName));
          handler.call(this, this[propertyName]);
        }, this);
      });

      this.model.on('change:other_options', () => {
        this.options = this.model.get('other_options');
        this.optionsHandler.call(this, this.options);
      }, this);

      this.colorCanvas();

      if (this.points.length) {
        this.scatterplot
          .draw(this.points)
          .then(() => {
            if (this.filter && this.filter.length) {
              this.scatterplot.filter(this.filter, { preventEvent: true });
              if (this.model.get('zoom_on_filter')) {
                this.zoomToHandler(this.filter);
              }
            }
            if (this.selection.length) {
              this.scatterplot.select(this.selection, { preventEvent: true });
              if (this.model.get('zoom_on_selection')) {
                this.zoomToHandler(this.selection);
              }
            }
          });
      }
    });

    this.model.save_changes();
  }

  getOuterDimensions() {
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
  }

  createAxes() {
    this.axesSvg = d3Selection.select(this.container).select('svg').node()
      ? d3Selection.select(this.container).select('svg')
      : d3Selection.select(this.container).append('svg');
    this.axesSvg.style('top', 0);
    this.axesSvg.style('left', 0);
    this.axesSvg.style('width', '100%');
    this.axesSvg.style('height', '100%');
    this.axesSvg.style('pointer-events', 'none');
    this.axesSvg.style('user-select', 'none');
    const color = this.model.get('axes_color').map((c) => Math.round(c * 255));
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
  }

  removeAxes() {
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
  }

  createAxesGrid() {
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
  }

  removeAxesGrid() {
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
  }

  showLegend() {
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
  }

  hideLegend() {
    if (!this.legendWrapper) return;
    this.container.removeChild(this.legendWrapper);
    this.legendWrapper = undefined;
    this.legend = undefined;
  }

  updateLegendWrapperPosition() {
    if (!this.legendWrapper) return;

    const labels = this.model.get('axes_labels');
    const xPadding = labels ? AXES_PADDING_X_WITH_LABEL : AXES_PADDING_X;
    const yPadding = labels ? AXES_PADDING_Y_WITH_LABEL : AXES_PADDING_Y;

    this.legendWrapper.style.top = 0;
    this.legendWrapper.style.bottom = yPadding + 'px';
    this.legendWrapper.style.left = 0;
    this.legendWrapper.style.right = xPadding + 'px';
  }

  updateLegendPosition() {
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
  }

  updateContainerDimensions() {
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
  }

  resizeHandler() {
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
  }

  destroy() {
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
    this.scatterplot.unsubscribe('filter', this.filterEventHandlerBound);
    this.scatterplot.unsubscribe('view', this.viewChangeHandlerBound);
    this.scatterplot.destroy();
  }

  // Helper
  colorCanvas() {
    if (Array.isArray(this.backgroundColor)) {
      this.container.style.backgroundColor = 'rgb(' +
        this.backgroundColor.slice(0, 3).map((x) => x * 255).join(',') + ')';
    } else {
      this.container.style.backgroundColor = this.backgroundColor;
    }
  }

  // Event handlers for JS-triggered events
  pointoverHandler(pointIndex) {
    this.hoveringChangedByJs = true;
    this.model.set('hovering', pointIndex);
    this.model.save_changes();
  }

  pointoutHandler() {
    this.hoveringChangedByJs = true;
    this.model.set('hovering', null);
    this.model.save_changes();
  }

  selectHandler(event) {
    this.selectionChangedByJs = true;
    if (this.model.get('zoom_on_selection')) this.zoomToHandler(event.points);
    this.model.set('selection', [...event.points]);
    this.model.save_changes();
  }

  deselectHandler() {
    this.selectionChangedByJs = true;
    if (this.model.get('zoom_on_selection')) this.zoomToHandler();
    this.model.set('selection', []);
    this.model.save_changes();
  }

  filterEventHandler(event) {
    this.filterChangedByJs = true;
    if (this.model.get('zoom_on_filter')) this.zoomToHandler(event.points);
    this.model.set('filter', [...event.points]);
    this.model.save_changes();
  }

  viewSyncHandler(viewSync) {
    if (viewSync) {
      pubSub.globalPubSub.subscribe(
        'jscatter::view', this.externalViewChangeHandlerBound
      );
    } else {
      pubSub.globalPubSub.unsubscribe(
        'jscatter::view', this.externalViewChangeHandlerBound
      );
    }
  }

  externalViewChangeHandler(event) {
    if (event.uuid === this.viewSync && event.src !== this.randomStr) {
      this.scatterplot.view(event.view, { preventEvent: true });
    }
  }

  viewChangeHandler(event) {
    if (this.viewSync) {
      pubSub.globalPubSub.publish(
        'jscatter::view',
        {
          src: this.randomStr,
          uuid: this.viewSync,
          view: event.view,
        },
        { async: true }
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
  }

  xScaleHandler() {
    if (this.model.get('axes')) this.createAxes();
  }

  yScaleHandler() {
    if (this.model.get('axes')) this.createAxes();
  }

  // Event handlers for Python-triggered events
  pointsHandler(newPoints) {
    if (newPoints.length === this.scatterplot.get('points').length) {
      // We assume point correspondence
      this.scatterplot.draw(newPoints, {
        transition: true,
        transitionDuration: 3000,
        transitionEasing: 'quadInOut',
      });
    } else {
      this.scatterplot.deselect();
      this.scatterplot.unfilter();
      this.scatterplot.draw(newPoints);
    }
  }

  selectionHandler(pointIdxs) {
    // Avoid calling `this.scatterplot.select()` twice when the selection was
    // triggered by the JavaScript (i.e., the user interactively selected points)
    if (this.selectionChangedByJs) {
      this.selectionChangedByJs = undefined;
      return;
    }

    const selection = pointIdxs?.length > 0
      ? pointIdxs
      : undefined;

    const options = { preventEvent: true };

    if (selection) this.scatterplot.select(selection, options);
    else this.scatterplot.deselect(options);

    if (this.model.get('zoom_on_selection')) this.zoomToHandler(selection);
  }

  filterHandler(pointIdxs) {
    // Avoid calling `this.scatterplot.select()` twice when the selection was
    // triggered by the JavaScript (i.e., the user interactively selected points)
    if (this.filterChangedByJs) {
      this.filterChangedByJs = undefined;
      return;
    }

    if (pointIdxs) {
      this.scatterplot.filter(pointIdxs, { preventEvent: true });
      if (this.model.get('zoom_on_filter')) this.zoomToHandler(pointIdxs);
    } else {
      this.scatterplot.unfilter({ preventEvent: true });
      if (this.model.get('zoom_on_filter')) this.zoomToHandler();
    }
  }

  hoveringHandler(newHovering) {
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
  }

  widthHandler() {
    this.updateContainerDimensions();
  }

  heightHandler() {
    this.updateContainerDimensions();
  }

  backgroundColorHandler(newValue) {
    this.withPropertyChangeHandler('backgroundColor', newValue);
    this.colorCanvas();
  }

  backgroundImageHandler(newValue) {
    this.withPropertyChangeHandler('backgroundImage', newValue);
  }

  lassoColorHandler(newValue) {
    this.withPropertyChangeHandler('lassoColor', newValue);
  }

  lassoMinDelayHandler(newValue) {
    this.withPropertyChangeHandler('lassoMinDelay', newValue);
  }

  lassoMinDistHandler(newValue) {
    this.withPropertyChangeHandler('lassoMinDist', newValue);
  }

  colorHandler(newValue) {
    this.withPropertyChangeHandler('pointColor', newValue);
  }

  colorSelectedHandler(newValue) {
    this.withPropertyChangeHandler('pointColorActive', newValue);
  }

  colorHoverHandler(newValue) {
    this.withPropertyChangeHandler('pointColorHover', newValue);
  }

  colorByHandler(newValue) {
    this.withPropertyChangeHandler('colorBy', newValue);
  }

  opacityHandler(newValue) {
    this.withPropertyChangeHandler('opacity', newValue);
  }

  opacityUnselectedHandler(newValue) {
    this.withPropertyChangeHandler('opacityInactiveScale', newValue);
  }

  opacityByHandler(newValue) {
    this.withPropertyChangeHandler('opacityBy', newValue);
  }

  sizeHandler(newValue) {
    this.withPropertyChangeHandler('pointSize', newValue);
  }

  sizeByHandler(newValue) {
    this.withPropertyChangeHandler('sizeBy', newValue);
  }

  connectHandler(newValue) {
    this.withPropertyChangeHandler('showPointConnections', Boolean(newValue));
  }

  connectionColorHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionColor', newValue);
  }

  connectionColorSelectedHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionColorActive', newValue);
  }

  connectionColorHoverHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionColorHover', newValue);
  }

  connectionColorByHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionColorBy', newValue);
  }

  connectionOpacityHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionOpacity', newValue);
  }

  connectionOpacityByHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionOpacityBy', newValue);
  }

  connectionSizeHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionSize', newValue);
  }

  connectionSizeByHandler(newValue) {
    this.withPropertyChangeHandler('pointConnectionSizeBy', newValue);
  }

  reticleHandler(newValue) {
    this.withPropertyChangeHandler('showReticle', newValue);
  }

  reticleColorHandler(newValue) {
    this.withPropertyChangeHandler('reticleColor', newValue);
  }

  cameraTargetHandler(newValue) {
    this.withPropertyChangeHandler('cameraTarget', newValue);
  }

  cameraDistanceHandler(newValue) {
    this.withPropertyChangeHandler('cameraDistance', newValue);
  }

  cameraRotationHandler(newValue) {
    this.withPropertyChangeHandler('cameraRotation', newValue);
  }

  cameraViewHandler(newValue) {
    this.withPropertyChangeHandler('cameraView', newValue);
  }

  lassoInitiatorHandler(newValue) {
    this.withPropertyChangeHandler('lassoInitiator', newValue);
  }

  lassoOnLongPressHandler(newValue) {
    this.withPropertyChangeHandler('lassoOnLongPress', newValue);
  }

  mouseModeHandler(newValue) {
    this.withPropertyChangeHandler('mouseMode', newValue);
  }

  axesHandler(newValue) {
    if (newValue) this.createAxes();
    else this.removeAxes();
  }

  axesColorHandler() {
    this.createAxes();
  }

  axesGridHandler(newValue) {
    if (newValue) this.createAxesGrid();
    else this.removeAxesGrid();
  }

  axesLabelsHandler(newValue) {
    if (!newValue) this.removeAxes();
    this.createAxes();
  }

  legendHandler(newValue) {
    if (newValue) this.showLegend();
    else this.hideLegend();
  }

  legendColorHandler() {
    this.hideLegend();
    this.showLegend();
  }

  legendSizeHandler() {
    this.hideLegend();
    this.showLegend();
  }

  legendPositionHandler() {
    this.updateLegendPosition();
  }

  legendEncodingHandler() {
    if (!this.model.get('legend')) return;
    this.showLegend();
  }

  zoomToHandler(points) {
    const animation = this.model.get('zoom_animation');
    const padding = this.model.get('zoom_padding');

    const transition = animation > 0;
    const transitionDuration = animation;

    const options = transition
      ? { padding, transition, transitionDuration }
      : { padding };

    if (points && points.length) {
      this.scatterplot.zoomToPoints(points, options);
    } else {
      this.scatterplot.zoomToOrigin(options);
    }
  }

  zoomToCallIdxHandler() {
    this.zoomToHandler(this.model.get('zoom_to'));
  }

  viewDownloadHandler(target) {
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
  }

  viewResetHandler() {
    this.scatterplot.reset();
    setTimeout(() => {
      this.model.set('view_reset', false);
      this.model.save_changes();
    }, 0);
  }

  optionsHandler(newOptions) {
    this.scatterplot.set(newOptions);
  }

  withPropertyChangeHandler(property, changedValue) {
    const p = {};
    p[property] = changedValue;
    this.scatterplot.set(p);
  }
};

function modelWithSerializers(model, serializers) {
  return {
    get(key) {
      const value = model.get(key);
      const serializer = serializers[key];
      if (serializer) return serializer.deserialize(value);
      return value;
    },
    set(key, value) {
      const serializer = serializers[key];
      if (serializer) value = serializer.serialize(value);
      model.set(key, value);
    },
    on: model.on.bind(model),
    save_changes: model.save_changes.bind(model),
  }
}

export async function render({ model, el }) {
  const view = new JupyterScatterView({
    el: el,
    model: modelWithSerializers(model, {
      points: codecs.Numpy2D('float32'),
      selection: codecs.Numpy1D('uint32'),
      filter: codecs.Numpy1D('uint32'),
      view_data: codecs.Numpy1D('uint8'),
      zoom_to: codecs.Numpy1D('uint32'),
    }),
  });
  view.render();
  return () => view.destroy();
}
