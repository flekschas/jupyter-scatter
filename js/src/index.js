import createScatterplot, { createRenderer } from 'regl-scatterplot';
import * as pubSub from 'pub-sub-es';
import { min, max, getD3FormatSpecifier } from '@flekschas/utils';

import { axisBottom, axisRight } from 'd3-axis';
import { format } from 'd3-format';
import { scaleLinear } from 'd3-scale';
import { select } from 'd3-selection';

import { Numpy1D, Numpy2D } from "./codecs";
import { createLegend } from "./legend";
import {
  camelToSnake,
  toCapitalCase,
  downloadBlob,
  getScale,
  createOrdinalScaleInverter,
  getTooltipFontSize,
} from "./utils";

import { version } from "../package.json";

const AXES_LABEL_SIZE = 16;
const AXES_PADDING_X = 40;
const AXES_PADDING_X_WITH_LABEL = AXES_PADDING_X + AXES_LABEL_SIZE;
const AXES_PADDING_Y = 20;
const AXES_PADDING_Y_WITH_LABEL = AXES_PADDING_Y + AXES_LABEL_SIZE;
const TOOLTIP_CONTENTS = ['x', 'y', 'color', 'opacity', 'size'];

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
  tooltipEnable: 'tooltipEnable',
  tooltipSize: 'tooltipSize',
  tooltipColor: 'tooltipColor',
  tooltipPosition: 'tooltipPosition',
  tooltipContents: 'tooltipContents',
  xScale: 'xScale',
  yScale: 'yScale',
  colorScale: 'colorScale',
  opacityScale: 'opacityScale',
  sizeScale: 'sizeScale',
  xDomain: 'xDomain',
  yDomain: 'yDomain',
  colorDomain: 'colorDomain',
  opacityDomain: 'opacityDomain',
  sizeDomain: 'sizeDomain',
  xTitle: 'xTitle',
  yTitle: 'yTitle',
  colorTitle: 'colorTitle',
  opacityTitle: 'opacityTitle',
  sizeTitle: 'sizeTitle',
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

      this.getOuterDimensions();
      this.createXScale();
      this.createYScale();
      this.createColorScale();
      this.createOpacityScale();
      this.createSizeScale();
      this.createXGetter();
      this.createYGetter();
      this.createColorGetter();
      this.createOpacityGetter();
      this.createSizeGetter();
      this.createTooltip();

      if (this.points.length) {
        const options = {}
        if (this.filter && this.filter.length) options.filter = this.filter;
        if (this.selection.length) options.select = this.selection;

        this.scatterplot
          .draw(this.points, options)
          .then(() => {
            if (this.filter?.length && this.model.get('zoom_on_filter')) {
              this.zoomToHandler(this.filter);
            }
            if (this.selection.length && this.model.get('zoom_on_selection')) {
              this.zoomToHandler(this.selection);
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

    this.outerWidth = outerWidth;
    this.outerHeight = outerHeight;

    return [outerWidth, outerHeight]
  }

  createAxes() {
    this.axesSvg = select(this.container).select('svg').node()
      ? select(this.container).select('svg')
      : select(this.container).append('svg');
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
    this.xScaleRegl = scaleLinear()
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

    this.yScaleRegl = scaleLinear()
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

    this.xAxis = axisBottom(this.xScaleAxis);
    this.yAxis = axisRight(this.yScaleAxis);

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

  createTooltip() {
    this.tooltip = document.createElement('div');
    this.tooltip.style.position = 'absolute';
    this.tooltip.style.top = 0;
    this.tooltip.style.left = 0;
    this.tooltip.style.pointerEvents = 'none';
    this.tooltip.style.userSelect = 'none';
    this.tooltip.style.borderRadius = '0.2rem';
    this.tooltip.style.opacity = 0;
    this.tooltip.style.transition = 'opacity 0.2s ease-in-out';

    this.tooltipArrow = document.createElement('div');
    this.tooltipArrow.style.position = 'absolute';
    this.tooltipArrow.style.width = '0.5rem';
    this.tooltipArrow.style.height = '0.5rem';
    this.tooltipArrow.style.transformOrigin = 'center';
    this.tooltip.appendChild(this.tooltipArrow);

    this.tooltipContent = document.createElement('div');
    this.tooltipContent.style.position = 'relative';
    this.tooltipContent.style.display = 'grid';
    this.tooltipContent.style.gridTemplateColumns = 'max-content max-content max-content';
    this.tooltipContent.style.gap = '0.3em 0.25em';
    this.tooltipContent.style.userSelect = 'none';
    this.tooltipContent.style.borderRadius = '0.2rem';
    this.tooltipContent.style.padding = '0.25em';
    this.tooltipContent.style.fontSize = getTooltipFontSize(this.model.get('tooltip_size'));
    this.tooltip.appendChild(this.tooltipContent);

    this.tooltipContentXChannel = document.createElement('div');
    this.tooltipContentXTitle = document.createElement('div');
    this.tooltipContentXValue = document.createElement('div');
    this.tooltipContentXValueBadge = document.createElement('div');
    this.tooltipContentXValueBadgeMark = document.createElement('div');
    this.tooltipContentXValueBadgeBg = document.createElement('div');
    this.tooltipContentXValueText = document.createElement('div');
    this.tooltipContentXValueBadge.appendChild(this.tooltipContentXValueBadgeMark);
    this.tooltipContentXValueBadge.appendChild(this.tooltipContentXValueBadgeBg);
    this.tooltipContentXValue.appendChild(this.tooltipContentXValueBadge);
    this.tooltipContentXValue.appendChild(this.tooltipContentXValueText);

    this.tooltipContentYChannel = document.createElement('div');
    this.tooltipContentYTitle = document.createElement('div');
    this.tooltipContentYValue = document.createElement('div');
    this.tooltipContentYValueBadge = document.createElement('div');
    this.tooltipContentYValueBadgeMark = document.createElement('div');
    this.tooltipContentYValueBadgeBg = document.createElement('div');
    this.tooltipContentYValueText = document.createElement('div');
    this.tooltipContentYValueBadge.appendChild(this.tooltipContentYValueBadgeMark);
    this.tooltipContentYValueBadge.appendChild(this.tooltipContentYValueBadgeBg);
    this.tooltipContentYValue.appendChild(this.tooltipContentYValueBadge);
    this.tooltipContentYValue.appendChild(this.tooltipContentYValueText);

    this.tooltipContentColorChannel = document.createElement('div');
    this.tooltipContentColorTitle = document.createElement('div');
    this.tooltipContentColorValue = document.createElement('div');
    this.tooltipContentColorValueBadge = document.createElement('div');
    this.tooltipContentColorValueBadgeMark = document.createElement('div');
    this.tooltipContentColorValueBadgeBg = document.createElement('div');
    this.tooltipContentColorValueText = document.createElement('div');
    this.tooltipContentColorValueBadge.appendChild(this.tooltipContentColorValueBadgeMark);
    this.tooltipContentColorValueBadge.appendChild(this.tooltipContentColorValueBadgeBg);
    this.tooltipContentColorValue.appendChild(this.tooltipContentColorValueBadge);
    this.tooltipContentColorValue.appendChild(this.tooltipContentColorValueText);

    this.tooltipContentOpacityChannel = document.createElement('div');
    this.tooltipContentOpacityTitle = document.createElement('div');
    this.tooltipContentOpacityValue = document.createElement('div');
    this.tooltipContentOpacityValueBadge = document.createElement('div');
    this.tooltipContentOpacityValueBadgeMark = document.createElement('div');
    this.tooltipContentOpacityValueBadgeBg = document.createElement('div');
    this.tooltipContentOpacityValueText = document.createElement('div');
    this.tooltipContentOpacityValueBadge.appendChild(this.tooltipContentOpacityValueBadgeMark);
    this.tooltipContentOpacityValueBadge.appendChild(this.tooltipContentOpacityValueBadgeBg);
    this.tooltipContentOpacityValue.appendChild(this.tooltipContentOpacityValueBadge);
    this.tooltipContentOpacityValue.appendChild(this.tooltipContentOpacityValueText);

    this.tooltipContentSizeChannel = document.createElement('div');
    this.tooltipContentSizeTitle = document.createElement('div');
    this.tooltipContentSizeValue = document.createElement('div');
    this.tooltipContentSizeValueBadge = document.createElement('div');
    this.tooltipContentSizeValueBadgeMark = document.createElement('div');
    this.tooltipContentSizeValueBadgeBg = document.createElement('div');
    this.tooltipContentSizeValueText = document.createElement('div');
    this.tooltipContentSizeValueBadge.appendChild(this.tooltipContentSizeValueBadgeMark);
    this.tooltipContentSizeValueBadge.appendChild(this.tooltipContentSizeValueBadgeBg);
    this.tooltipContentSizeValue.appendChild(this.tooltipContentSizeValueBadge);
    this.tooltipContentSizeValue.appendChild(this.tooltipContentSizeValueText);

    this.tooltipContentXChannel.textContent = 'X';
    this.tooltipContentYChannel.textContent = 'Y';
    this.tooltipContentColorChannel.textContent = 'Col';
    this.tooltipContentOpacityChannel.textContent = 'Opa';
    this.tooltipContentSizeChannel.textContent = 'Size';

    this.tooltipContentXTitle.textContent = toCapitalCase(this.model.get('x_title') || '');
    this.tooltipContentYTitle.textContent = toCapitalCase(this.model.get('y_title') || '');
    this.tooltipContentColorTitle.textContent = toCapitalCase(this.model.get('color_title') || '');
    this.tooltipContentOpacityTitle.textContent = toCapitalCase(this.model.get('opacity_title') || '');
    this.tooltipContentSizeTitle.textContent = toCapitalCase(this.model.get('size_title') || '');

    [
      this.tooltipContentXChannel,
      this.tooltipContentXTitle,
      this.tooltipContentXValue,
      this.tooltipContentYChannel,
      this.tooltipContentYTitle,
      this.tooltipContentYValue,
      this.tooltipContentColorChannel,
      this.tooltipContentColorTitle,
      this.tooltipContentColorValue,
      this.tooltipContentOpacityChannel,
      this.tooltipContentOpacityTitle,
      this.tooltipContentOpacityValue,
      this.tooltipContentSizeChannel,
      this.tooltipContentSizeTitle,
      this.tooltipContentSizeValue,
    ].forEach((el) => this.tooltipContent.appendChild(el));

    this.styleTooltip();
    this.enableTooltipContents();
    this.createTooltipContentUpdater();

    this.container.appendChild(this.tooltip);
  }

  positionTooltipTopCenter() {
    this.tooltipArrow.style.removeProperty('top');
    this.tooltipArrow.style.removeProperty('right');
    this.tooltipArrow.style.bottom = 0;
    this.tooltipArrow.style.left = '50%';
    this.tooltipArrow.style.transform = 'translate(-50%, calc(50% - 1px)) rotate(-45deg)';
    this.tooltipArrow.style.boxShadow = `0 0 1px rgba(0, 0, 0, ${this.tooltipOpacity}), -1px 1px 2px rgba(0, 0, 0, ${this.tooltipOpacity})`;

    this.moveTooltip = (x, y) => {
      this.tooltip.style.transform = `translate(calc(${x}px - 50%), calc(${y}px - 0.5rem - 100%))`;
    }
  }

  positionTooltipTopLeft() {
    this.tooltipArrow.style.removeProperty('top');
    this.tooltipArrow.style.right = '0.125rem';
    this.tooltipArrow.style.bottom = 0;
    this.tooltipArrow.style.removeProperty('left');
    this.tooltipArrow.style.transform = 'translate(-50%, calc(50% - 1px)) rotate(-45deg)';
    this.tooltipArrow.style.boxShadow = `0 0 1px rgba(0, 0, 0, ${this.tooltipOpacity}), -1px 1px 2px rgba(0, 0, 0, ${this.tooltipOpacity})`;

    this.moveTooltip = (x, y) => {
      this.tooltip.style.transform = `translate(calc(${x}px - 100% + 0.625rem), calc(${y}px - 0.5rem - 100%))`;
    }
  }

  positionTooltipTopRight() {
    this.tooltipArrow.style.removeProperty('top');
    this.tooltipArrow.style.removeProperty('right');
    this.tooltipArrow.style.bottom = 0;
    this.tooltipArrow.style.left = '0.5rem';
    this.tooltipArrow.style.transform = 'translate(-50%, calc(50% - 1px)) rotate(-45deg)';
    this.tooltipArrow.style.boxShadow = `0 0 1px rgba(0, 0, 0, ${this.tooltipOpacity}), -1px 1px 2px rgba(0, 0, 0, ${this.tooltipOpacity})`;

    this.moveTooltip = (x, y) => {
      this.tooltip.style.transform = `translate(calc(${x}px - 0.5rem), calc(${y}px - 0.5rem - 100%))`;
    }
  }

  positionTooltipBottomCenter() {
    this.tooltipArrow.style.top = 0;
    this.tooltipArrow.style.removeProperty('right');
    this.tooltipArrow.style.removeProperty('bottom');
    this.tooltipArrow.style.left = '50%';
    this.tooltipArrow.style.transform = 'translate(-50%, calc(-50% + 1px)) rotate(-45deg)';
    this.tooltipArrow.style.boxShadow = `0 0 1px rgba(0, 0, 0, ${this.tooltipOpacity}), -1px -1px 2px rgba(0, 0, 0, ${this.tooltipOpacity})`;

    this.moveTooltip = (x, y) => {
      this.tooltip.style.transform = `translate(calc(${x}px - 50%), calc(${y}px + 0.5rem))`;
    }
  }

  positionTooltipBottomLeft() {
    this.tooltipArrow.style.top = 0;
    this.tooltipArrow.style.right = '0.125rem';
    this.tooltipArrow.style.removeProperty('bottom');
    this.tooltipArrow.style.removeProperty('left');
    this.tooltipArrow.style.transform = 'translate(-50%, calc(-50% + 1px)) rotate(-45deg)';
    this.tooltipArrow.style.boxShadow = `0 0 1px rgba(0, 0, 0, ${this.tooltipOpacity}), -1px -1px 2px rgba(0, 0, 0, ${this.tooltipOpacity})`;

    this.moveTooltip = (x, y) => {
      this.tooltip.style.transform = `translate(calc(${x}px - 100% + 0.625rem), calc(${y}px + 0.5rem))`;
    }
  }

  positionTooltipBottomRight() {
    this.tooltipArrow.style.top = 0;
    this.tooltipArrow.style.removeProperty('right');
    this.tooltipArrow.style.removeProperty('bottom');
    this.tooltipArrow.style.left = '0.5rem';
    this.tooltipArrow.style.transform = 'translate(-50%, calc(-50% + 1px)) rotate(-45deg)';
    this.tooltipArrow.style.boxShadow = `0 0 1px rgba(0, 0, 0, ${this.tooltipOpacity}), -1px -1px 2px rgba(0, 0, 0, ${this.tooltipOpacity})`;

    this.moveTooltip = (x, y) => {
      this.tooltip.style.transform = `translate(calc(${x}px - 0.5rem), calc(${y}px + 0.5rem))`;
    }
  }

  positionTooltipLeftCenter() {
    this.tooltipArrow.style.top = '50%';
    this.tooltipArrow.style.right = 0;
    this.tooltipArrow.style.removeProperty('bottom');
    this.tooltipArrow.style.removeProperty('left');
    this.tooltipArrow.style.transform = 'translate(calc(50% - 1px), -50%) rotate(-45deg)';
    this.tooltipArrow.style.boxShadow = `0 0 1px rgba(0, 0, 0, ${this.tooltipOpacity}), -1px -1px 2px rgba(0, 0, 0, ${this.tooltipOpacity})`;

    this.moveTooltip = (x, y) => {
      this.tooltip.style.transform = `translate(calc(${x}px - 0.5rem - 100%), calc(${y}px - 50%))`;
    }
  }

  positionTooltipRightCenter() {
    this.tooltipArrow.style.top = '50%';
    this.tooltipArrow.style.removeProperty('right');
    this.tooltipArrow.style.removeProperty('bottom');
    this.tooltipArrow.style.left = 0;
    this.tooltipArrow.style.transform = 'translate(calc(-50% + 1px), -50%) rotate(-45deg)';
    this.tooltipArrow.style.boxShadow = `0 0 1px rgba(0, 0, 0, ${this.tooltipOpacity}), 1px 1px 2px rgba(0, 0, 0, ${this.tooltipOpacity})`;
    this.moveTooltip = (x, y) => {
      this.tooltip.style.transform = `translate(calc(${x}px + 0.5rem), calc(${y}px - 50%))`;
    }
  }

  positionTooltip(position) {
    switch (position) {
      case 'bottom-center':
        return this.positionTooltipBottomCenter();
      case 'bottom-left':
        return this.positionTooltipBottomLeft();
      case 'bottom-right':
        return this.positionTooltipBottomRight();
      case 'left-center':
        return this.positionTooltipLeftCenter();
      case 'right-center':
        return this.positionTooltipRightCenter();
      case 'top-left':
        return this.positionTooltipTopLeft();
      case 'top-right':
        return this.positionTooltipTopRight();
      default:
        this.positionTooltipTopCenter();
    }
  }

  getTooltipPosition(x, y) {
    if (x < 120) {
      if (y < 120) return 'bottom-right';
      if (y > this.outerHeight - 120) return 'top-right';
      return 'right-center';
    }

    if (x > this.outerWidth - 120) {
      if (y < 120) return 'bottom-left';
      if (y > this.outerHeight - 120) return 'top-left';
      return 'left-center';
    }

    if (y < 120) return 'bottom-center';

    return 'top-center';
  }

  isTooltipContentShown(content) {
    if (!this.tooltipContents.has(content)) return false;
    if (content === 'x') return true;
    if (content === 'y') return true;
    if (content === 'color') return Boolean(this.model.get('color_by'));
    if (content === 'opacity') return this.model.get('opacity_by') && this.model.get('opacity_by') !== 'density';
    if (content === 'size') return Boolean(this.model.get('size_by'));
  }

  enableTooltipContents() {
    this.tooltipContents = new Set(this.model.get('tooltip_contents'));
    for (const content of TOOLTIP_CONTENTS) {
      const title = toCapitalCase(content);
      const display = this.isTooltipContentShown(content) ? 'flex' : 'none';

      this[`tooltipContent${title}Channel`].style.display = display;
      this[`tooltipContent${title}Title`].style.display = display;
      this[`tooltipContent${title}Value`].style.display = display;
    }
  }

  styleTooltip() {
    if (!this.tooltip) this.createTooltip();
    const color = this.model.get('tooltip_color').map((c) => Math.round(c * 255));

    const isDark = color[0] <= 127;
    const contrast = isDark ? 255 : 0;
    const bg = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

    this.tooltipOpacity = isDark ? 0.33 : 0.2;

    this.tooltip.style.opacity = 0;
    this.tooltip.style.color = `rgb(${contrast}, ${contrast}, ${contrast})`;
    this.tooltip.style.backgroundColor = bg;
    this.tooltip.style.boxShadow = `0 0 1px rgba(0, 0, 0, ${this.tooltipOpacity}), 0 1px 2px rgba(0, 0, 0, ${this.tooltipOpacity})`;
    this.tooltipArrow.style.backgroundColor = bg;
    this.tooltipContent.style.backgroundColor = bg;

    this.tooltipContentXValue.style.alignItems = 'center';
    this.tooltipContentYValue.style.alignItems = 'center';
    this.tooltipContentColorValue.style.alignItems = 'center';
    this.tooltipContentOpacityValue.style.alignItems = 'center';
    this.tooltipContentSizeValue.style.alignItems = 'center';

    [
      this.tooltipContentXValueBadge,
      this.tooltipContentYValueBadge,
      this.tooltipContentColorValueBadge,
      this.tooltipContentOpacityValueBadge,
      this.tooltipContentSizeValueBadge,
    ].forEach((element) => {
      element.style.position = 'relative';
      element.style.width = '1em';
      element.style.height = '1em';
      element.style.marginRight = '0.125rem';
    });

    [
      this.tooltipContentXValueBadgeBg,
      this.tooltipContentYValueBadgeBg,
      this.tooltipContentColorValueBadgeBg,
      this.tooltipContentOpacityValueBadgeBg,
      this.tooltipContentSizeValueBadgeBg,
    ].forEach((element) => {
      element.style.position = 'absolute';
      element.style.zIndex = 2;
      element.style.top = 0;
      element.style.left = 0;
      element.style.width = '1em';
      element.style.height = '1em';
    });

    this.tooltipContentXValueBadgeBg.style.zIndex = 0;
    this.tooltipContentXValueBadgeBg.style.top = '50%';
    this.tooltipContentXValueBadgeBg.style.transform = 'translate(0, -50%)';
    this.tooltipContentXValueBadgeBg.style.height = '2px';
    this.tooltipContentXValueBadgeBg.style.background = `rgba(${contrast}, ${contrast}, ${contrast}, 0.2)`;

    this.tooltipContentXValueBadgeMark.style.position = 'absolute';
    this.tooltipContentXValueBadgeMark.style.zIndex = 1;
    this.tooltipContentXValueBadgeMark.style.top = '50%';
    this.tooltipContentXValueBadgeMark.style.width = '2px';
    this.tooltipContentXValueBadgeMark.style.height = '6px';
    this.tooltipContentXValueBadgeMark.style.background = `rgb(${contrast}, ${contrast}, ${contrast})`;

    this.tooltipContentYValueBadgeBg.style.zIndex = 0;
    this.tooltipContentYValueBadgeBg.style.left = '50%';
    this.tooltipContentYValueBadgeBg.style.transform = 'translate(-50%, 0)';
    this.tooltipContentYValueBadgeBg.style.width = '2px';
    this.tooltipContentYValueBadgeBg.style.background = `rgba(${contrast}, ${contrast}, ${contrast}, 0.2)`;

    this.tooltipContentYValueBadgeMark.style.position = 'absolute';
    this.tooltipContentYValueBadgeMark.style.zIndex = 1;
    this.tooltipContentYValueBadgeMark.style.left = '50%';
    this.tooltipContentYValueBadgeMark.style.width = '6px';
    this.tooltipContentYValueBadgeMark.style.height = '2px';
    this.tooltipContentYValueBadgeMark.style.background = `rgb(${contrast}, ${contrast}, ${contrast})`;

    [
      this.tooltipContentOpacityValueBadgeBg,
      this.tooltipContentSizeValueBadgeBg,
    ].forEach((element) => {
      element.style.borderRadius = '1em';
      element.style.boxShadow = `inset 0 0 0 1px rgba(${contrast}, ${contrast}, ${contrast}, 0.2)`;
    });

    [
      this.tooltipContentColorValueBadgeMark,
      this.tooltipContentOpacityValueBadgeMark,
      this.tooltipContentSizeValueBadgeMark,
    ].forEach((element) => {
      element.style.width = '1em';
      element.style.height = '1em';
      element.style.borderRadius = '1em';
    });

    [
      this.tooltipContentOpacityValueBadgeMark,
      this.tooltipContentSizeValueBadgeMark,
    ].forEach((element) => {
      element.style.background = `rgb(${contrast}, ${contrast}, ${contrast})`;
    });

    const channelBg = `rgba(${contrast}, ${contrast}, ${contrast}, 0.075)`;
    const channelColor = `rgba(${contrast}, ${contrast}, ${contrast}, 0.5)`;

    [
      this.tooltipContentXChannel,
      this.tooltipContentYChannel,
      this.tooltipContentColorChannel,
      this.tooltipContentOpacityChannel,
      this.tooltipContentSizeChannel,
    ].forEach((el) => {
      el.style.display = 'flex';
      el.style.justifyContent = 'center';
      el.style.alignItems = 'center';
      el.style.padding = '0 0.125rem';
      el.style.borderRadius = '0.125rem';
      el.style.color = channelColor;
      el.style.fontSize = '0.8em';
      el.style.fontWeight = 'bold';
      el.style.textTransform = 'uppercase';
      el.style.background = channelBg;
    });

    [
      this.tooltipContentXValue,
      this.tooltipContentYValue,
      this.tooltipContentColorValue,
      this.tooltipContentOpacityValue,
      this.tooltipContentSizeValue,
    ].forEach((el) => {
      el.style.fontWeight = 'bold';
    });


  }

  getPoint(i) {
    return this.scatterplot.get('points')[i];
  }

  getPoints() {
    return this.scatterplot.get('points');
  }

  createXGetter() {
    if (!this.xScale) this.createXScale();
    this.getX = (i) => {
      const xNdc = this.getPoint(i)[0];
      return [(xNdc + 1) / 2, this.xFormat(this.xScale.invert(xNdc))];
    }
  }

  createYGetter() {
    if (!this.yScale) this.createYScale();
    this.getY = (i) => {
      const yNdc = this.getPoint(i)[1];
      return [(yNdc + 1) / 2, this.yFormat(this.yScale.invert(yNdc))];
    }
  }

  createColorGetter() {
    if (!this.colorScale) this.createColorScale();
    if (!this.colorScale) {
      this.getColor = () => ['#808080', 'Unknown'];
      return;
    }

    const dim = this.model.get('color_by') === 'valueA' ? 2 : 3;
    const colors = this.model.get('color').map((color) => `rgb(${color.slice(0, 3).map((v) => Math.round(v * 255)).join(', ')})`);

    if (this.model.get('color_scale') === 'categorical') {
      this.getColor = (i) => {
        const value = this.getPoint(i)[dim];
        return [
          colors[value] || '#808080',
          this.colorFormat(this.colorScale.invert(value)),
        ]
      }
    } else {
      const numColors = colors.length;
      this.getColor = (i) => {
        const value = this.getPoint(i)[dim];
        const colorIdx = Math.min(numColors - 1, Math.floor(numColors * value));
        return [
          colors[colorIdx] || '#808080',
          this.colorFormat(this.colorScale.invert(value)),
        ]
      }
    }
  }

  createOpacityGetter() {
    if (!this.opacityScale) this.createOpacityScale();
    if (!this.opacityScale) {
      this.getOpacity = () => [0.5, 'Unknown'];
      return;
    }

    const dim = this.model.get('opacity_by') === 'valueA' ? 2 : 3;
    const opacities = this.model.get('opacity');

    if (this.model.get('opacity_scale') === 'categorical') {
      this.getOpacity = (i) => {
        const value = this.getPoint(i)[dim];
        return [
          opacities[value] || '#808080',
          this.opacityFormat(this.opacityScale.invert(value)),
        ]
      }
    } else {
      const numOpacities = opacities.length;
      this.getOpacity = (i) => {
        const value = this.getPoint(i)[dim];
        const idx = Math.min(numOpacities - 1, Math.floor(numOpacities * value));
        return [
          opacities[idx] || 0.5,
          this.opacityFormat(this.opacityScale.invert(value)),
        ]
      }
    }
  }

  createSizeGetter() {
    if (!this.sizeScale) this.createSizeScale();
    if (!this.sizeScale) {
      this.getSize = () => [0.5, 'Unknown'];
      return;
    }

    const dim = this.model.get('size_by') === 'valueA' ? 2 : 3;
    const sizes = this.model.get('size');
    const sizesMin = min(sizes);
    const sizesMax = max(sizes);
    const sizesExtent = sizesMax - sizesMin;

    if (this.model.get('size_scale') === 'categorical') {
      this.getSize = (i) => {
        const value = this.getPoint(i)[dim];
        return [
          sizes[value] !== undefined
            ? Math.max(0.1, (sizes[value] - sizesMin) / sizesExtent)
            : '#808080',
          this.sizeFormat(this.sizeScale.invert(value)),
        ]
      }
    } else {
      const numSizes = sizes.length;
      this.getSize = (i) => {
        const value = this.getPoint(i)[dim];
        const idx = Math.min(numSizes - 1, Math.floor(numSizes * value));
        return [
          Math.max(0.1, (sizes[idx] - sizesMin) / sizesExtent),
          this.sizeFormat(this.sizeScale.invert(value)),
        ]
      }
    }
  }

  createTooltipContentUpdater() {
    const contents = new Set(this.model.get('tooltip_contents'));
    const updaters = Array.from(contents).map((content) => {
      const contentTitle = toCapitalCase(content);
      const get = this[`get${contentTitle}`];

      const textElement = this[`tooltipContent${contentTitle}ValueText`];
      const badgeElement = this[`tooltipContent${contentTitle}ValueBadgeMark`];

      if (content === 'x') {
        return (pointIdx) => {
          const [x, text] = get(pointIdx);
          badgeElement.style.transform = `translate(calc(${x}em - 50%), -50%)`;
          textElement.textContent = text;
        }
      }

      if (content === 'y') {
        return (pointIdx) => {
          const [y, text] = get(pointIdx);
          badgeElement.style.transform = `translate(-50%, calc(${1 - y}em - 50%))`;
          textElement.textContent = text;
        }
      }

      if (content === 'color') {
        return (pointIdx) => {
          const [color, text] = get(pointIdx);
          badgeElement.style.background = color;
          textElement.textContent = text;
        }
      }

      if (content === 'opacity') {
        return (pointIdx) => {
          const [opacity, text] = get(pointIdx);
          badgeElement.style.opacity = opacity;
          textElement.textContent = text;
        }
      }

      if (content === 'size') {
        return (pointIdx) => {
          const [scale, text] = get(pointIdx);
          badgeElement.style.transform = `scale(${scale})`;
          textElement.textContent = text;
        }
      }

      return (pointIdx) => textElement.textContent = get(pointIdx);
    });
    this.tooltipContentsUpdater = (pointIdx) => {
      updaters.forEach((updater) => { updater(pointIdx); });
    }
  }

  showToolip(pointIdx) {
    const [x, y] = this.scatterplot.getScreenPosition(pointIdx);

    const newTooltipPosition = this.getTooltipPosition(x, y);
    if (newTooltipPosition !== this.tooltipPosition) {
      this.tooltipPosition = newTooltipPosition;
      this.positionTooltip(this.tooltipPosition);
    }

    this.moveTooltip(x, y);
    this.tooltip.style.opacity = 1;
    this.tooltipContentsUpdater(pointIdx);
  }

  hideToolip() {
    this.tooltip.style.opacity = 0;
  }

  tooltipEnableHandler(tooltip) {
    if (!tooltip) this.hideToolip;
  }

  tooltipSizeHandler(newSize) {
    this.tooltipContent.style.fontSize = getTooltipFontSize(newSize);
  }

  tooltipColorHandler() {
    this.styleTooltip();
  }

  tooltipContentsHandler() {
    this.enableTooltipContents();
    this.createTooltipContentUpdater();
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
    if (this.model.get('tooltip_enable')) this.showToolip(pointIndex);
    this.model.set('hovering', pointIndex);
    this.model.save_changes();
  }

  pointoutHandler() {
    this.hoveringChangedByJs = true;
    this.hideToolip();
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

  updateAxes(xScaleDomain, yScaleDomain) {
    if (!this.model.get('axes')) return;

    this.xScaleAxis.domain(xScaleDomain.map(this.xScaleRegl2Axis.invert));
    this.yScaleAxis.domain(yScaleDomain.map(this.yScaleRegl2Axis.invert));

    this.xAxisContainer.call(this.xAxis.scale(this.xScaleAxis));
    this.yAxisContainer.call(this.yAxis.scale(this.yScaleAxis));

    if (this.model.get('axes_grid')) {
      this.axesSvg.selectAll('line')
        .attr('stroke-opacity', 0.2)
        .attr('stroke-dasharray', 2);
    }
  }

  externalViewChangeHandler(event) {
    if (event.uuid === this.viewSync && event.src !== this.randomStr) {
      this.scatterplot.view(event.view, { preventEvent: true });
      if (this.model.get('axes')) {
        this.updateAxes(event.xScaleDomain, event.yScaleDomain);
      }
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
          xScaleDomain: event.xScale.domain(),
          yScaleDomain: event.yScale.domain(),
        },
        { async: true }
      );
    }
    if (this.model.get('axes')) {
      this.updateAxes(event.xScale.domain(), event.yScale.domain());
    }
  }

  createXScale() {
    this.xScale = getScale(this.model.get('x_scale'))
      .domain(this.model.get('x_domain'))
      .range([-1, 1]);
    this.xFormat = format(getD3FormatSpecifier(this.model.get('x_domain')));
  }

  createYScale() {
    this.yScale = getScale(this.model.get('y_scale'))
      .domain(this.model.get('y_domain'))
      .range([-1, 1]);
    this.yFormat = format(getD3FormatSpecifier(this.model.get('y_domain')));
  }

  createColorScale() {
    if (this.model.get('color_by')) {
      const scaleType = this.model.get('color_scale');
      const scale = getScale(scaleType);
      const domain = this.model.get('color_domain');

      if (scaleType === 'categorical') {
        this.colorScale = scale
          .domain(Object.keys(domain))
          .range(Object.values(domain));
        this.colorScale.invert = createOrdinalScaleInverter(domain);
        this.colorFormat = (s) => s;
      } else {
        this.colorScale = scale
          .domain(domain)
          .range([0, 1]);
        this.colorFormat = format(getD3FormatSpecifier(domain));
      }
    } else {
      this.colorScale = undefined;
      this.colorFormat = undefined;
    }
  }

  createOpacityScale() {
    const opacityBy = this.model.get('opacity_by');
    if (opacityBy && opacityBy !== 'density') {
      const scaleType = this.model.get('opacity_scale');
      const scale = getScale(scaleType);
      const domain = this.model.get('opacity_domain');

      if (scaleType === 'categorical') {
        this.opacityScale = scale
          .domain(Object.keys(domain))
          .range(Object.values(domain));
        this.opacityScale.invert = createOrdinalScaleInverter(domain);
        this.opacityFormat = (s) => s;
      } else {
        this.opacityScale = scale
          .domain(domain)
          .range([0, 1]);
        this.opacityFormat = format(getD3FormatSpecifier(domain));
      }
    } else {
      this.opacityScale = undefined;
      this.opacityFormat = undefined;
    }
  }

  createSizeScale() {
    if (this.model.get('size_by')) {
      const scaleType = this.model.get('size_scale');
      const scale = getScale(scaleType);
      const domain = this.model.get('size_domain');

      if (scaleType === 'categorical') {
        this.sizeScale = scale
          .domain(Object.keys(domain))
          .range(Object.values(domain));
        this.sizeScale.invert = createOrdinalScaleInverter(domain);
        this.sizeFormat = (s) => s;
      } else {
        this.sizeScale = scale
          .domain(domain)
          .range([0, 1]);
        this.sizeFormat = format(getD3FormatSpecifier(domain));
      }
    } else {
      this.sizeScale = undefined;
      this.sizeFormat = undefined;
    }
  }

  xScaleHandler() {
    this.createXScale();
    if (this.model.get('axes')) this.createAxes();
  }

  yScaleHandler() {
    this.createYScale();
    if (this.model.get('axes')) this.createAxes();
  }

  xDomainHandler() {
    this.createXScale();
    if (this.model.get('axes')) this.createAxes();
  }

  yDomainHandler() {
    this.createYScale();
    if (this.model.get('axes')) this.createAxes();
  }

  colorDomainHandler() {
    this.createColorScale();
  }

  opacityDomainHandler() {
    this.createOpacityScale();
  }

  sizeDomainHandler() {
    this.createSizeScale();
  }

  // Event handlers for Python-triggered events
  pointsHandler(newPoints) {
    if (newPoints.length === this.scatterplot.get('points').length) {
      // We assume point correspondence
      this.scatterplot.draw(newPoints, {
        transition: true,
        transitionDuration: 3000,
        transitionEasing: 'quadInOut',
        preventFilterReset: this.model.get('prevent_filter_reset'),
      });
    } else {
      this.scatterplot.deselect();
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

  xTitleHandler(newTitle) {
    this.tooltipContentXTitle.textContent = toCapitalCase(newTitle || '');
  }

  yTitleHandler(newTitle) {
    this.tooltipContentYTitle.textContent = toCapitalCase(newTitle || '');
  }

  colorHandler(newValue) {
    this.createColorScale();
    this.createColorGetter();
    this.withPropertyChangeHandler('pointColor', newValue);
  }

  colorSelectedHandler(newValue) {
    this.withPropertyChangeHandler('pointColorActive', newValue);
  }

  colorHoverHandler(newValue) {
    this.withPropertyChangeHandler('pointColorHover', newValue);
  }

  colorByHandler(newValue) {
    this.createColorScale();
    this.createColorGetter();
    this.withPropertyChangeHandler('colorBy', newValue);
  }

  colorTitleHandler(newTitle) {
    this.tooltipContentColorTitle.textContent = toCapitalCase(newTitle || '');
  }

  opacityHandler(newValue) {
    this.createOpacityScale();
    this.createOpacityGetter();
    this.withPropertyChangeHandler('opacity', newValue);
  }

  opacityUnselectedHandler(newValue) {
    this.withPropertyChangeHandler('opacityInactiveScale', newValue);
  }

  opacityByHandler(newValue) {
    // this.createOpacityScale();
    // this.createOpacityGetter();
    this.withPropertyChangeHandler('opacityBy', newValue);
  }

  opacityTitleHandler(newTitle) {
    this.tooltipContentOpacityTitle.textContent = toCapitalCase(newTitle || '');
  }

  sizeHandler(newValue) {
    this.createSizeScale();
    this.createSizeGetter();
    this.withPropertyChangeHandler('pointSize', newValue);
  }

  sizeByHandler(newValue) {
    this.createSizeScale();
    this.createSizeGetter();
    this.withPropertyChangeHandler('sizeBy', newValue);
  }

  sizeTitleHandler(newTitle) {
    this.tooltipContentSizeTitle.textContent = toCapitalCase(newTitle || '');
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
      points: Numpy2D('float32'),
      selection: Numpy1D('uint32'),
      filter: Numpy1D('uint32'),
      view_data: Numpy1D('uint8'),
      zoom_to: Numpy1D('uint32'),
    }),
  });
  view.render();
  return () => view.destroy();
}
