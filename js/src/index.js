import createScatterplot, { createRenderer } from 'regl-scatterplot';
import * as pubSub from 'pub-sub-es';
import { debounce, min, max, getD3FormatSpecifier } from '@flekschas/utils';

import { axisBottom, axisRight } from 'd3-axis';
import { format } from 'd3-format';
import { scaleLinear } from 'd3-scale';
import { select } from 'd3-selection';

import { Annotations, Numpy1D, Numpy2D, NumpyImage } from "./codecs";
import { createHistogram } from "./histogram";
import { createLegend } from "./legend";
import {
  camelToSnake,
  toCapitalCase,
  toHtmlClass,
  toTitleCase,
  downloadBlob,
  getScale,
  createOrdinalScaleInverter,
  getTooltipFontSize,
  createNumericalBinGetter,
  createElementWithClass,
  remToPx,
  createTimeFormat,
  addBackgroundColor,
  imageDataToCanvas,
} from "./utils";

import { version } from "../package.json";

const AXES_LABEL_SIZE = 12;
const AXES_PADDING_X = 60;
const AXES_PADDING_X_WITH_LABEL = AXES_PADDING_X + AXES_LABEL_SIZE;
const AXES_PADDING_Y = 20;
const AXES_PADDING_Y_WITH_LABEL = AXES_PADDING_Y + AXES_LABEL_SIZE;
const TOOLTIP_DEBOUNCE_TIME = 250;
const TOOLTIP_MANDATORY_VISUAL_PROPERTIES = (/** @type {const} */ ({ x: 'X', y: 'Y' }));
const TOOLTIP_OPTIONAL_VISUAL_PROPERTIES = (/** @type {const} */ ({ color: 'Col', opacity: 'Opa', size: 'Size' }));
const TOOLTIP_ALL_VISUAL_PROPERTIES = (/** @type {const} */ ({
  ...TOOLTIP_MANDATORY_VISUAL_PROPERTIES,
  ...TOOLTIP_OPTIONAL_VISUAL_PROPERTIES
}));
const TOOLTIP_HISTOGRAM_WIDTH = (/** @type {const} */ ({
  small: '6em',
  medium: '10em',
  large: '14em',
}));
const TOOLTIP_HISTOGRAM_HEIGHT = (/** @type {const} */ ({
  small: '1em',
  medium: '1.25em',
  large: '1.5em',
}));
const TOOLTIP_OFFSET_REM = 0.5;

/**
 * This dictionary maps between the camelCased Python property names and their
 * JavaScript counter parts. In most cases the name is identical but they can be
 * different. E.g., size (Python) vs pointSize (JavaScript)
 */
const properties = {
  annotations: 'annotations',
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
  reglScatterplotOptions: 'reglScatterplotOptions',
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
  tooltipProperties: 'tooltipProperties',
  tooltipHistograms: 'tooltipHistograms',
  tooltipHistogramsRanges: 'tooltipHistogramsRanges',
  tooltipHistogramsSize: 'tooltipHistogramsSize',
  tooltipPropertiesNonVisualInfo: 'tooltipPropertiesNonVisualInfo',
  tooltipPreview: 'tooltipPreview',
  tooltipPreviewType: 'tooltipPreviewType',
  tooltipPreviewTextLines: 'tooltipPreviewTextLines',
  tooltipPreviewTextMarkdown: 'tooltipPreviewTextMarkdown',
  tooltipPreviewImagePosition: 'tooltipPreviewImagePosition',
  tooltipPreviewImageSize: 'tooltipPreviewImageSize',
  tooltipPreviewAudioLength: 'tooltipPreviewAudioLength',
  tooltipPreviewAudioLoop: 'tooltipPreviewAudioLoop',
  xScale: 'xScale',
  yScale: 'yScale',
  colorScale: 'colorScale',
  opacityScale: 'opacityScale',
  sizeScale: 'sizeScale',
  xDomain: 'xDomain',
  yDomain: 'yDomain',
  xScaleDomain: 'xScaleDomain',
  yScaleDomain: 'yScaleDomain',
  colorDomain: 'colorDomain',
  opacityDomain: 'opacityDomain',
  sizeDomain: 'sizeDomain',
  xHistogram: 'xHistogram',
  yHistogram: 'yHistogram',
  colorHistogram: 'colorHistogram',
  opacityHistogram: 'opacityHistogram',
  sizeHistogram: 'sizeHistogram',
  xHistogramRange: 'xHistogramRange',
  yHistogramRange: 'yHistogramRange',
  colorHistogramRange: 'colorHistogramRange',
  opacityHistogramRange: 'opacityHistogramRange',
  sizeHistogramRange: 'sizeHistogramRange',
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
    this.eventTypes = model.get('event_types');
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
    this.canvasWrapper.style.inset = '0';
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
        if (this[pyName] !== null && reglScatterplotProperty.has(jsName)) {
          initialOptions[jsName] = this[pyName];
        }
        if (this[pyName] !== null && jsName === 'reglScatterplotOptions') {
          Object.entries(this[pyName]).forEach(([key, value]) => {
            initialOptions[key] = value;
          })
        }
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

      this.viewSync = this.model.get('view_sync');
      this.viewSyncHandler(this.viewSync);

      if ('ResizeObserver' in window) {
        this.canvasObserver = new ResizeObserver(() => {
          window.requestAnimationFrame(() => { this.resizeHandlerBound(); });
        });
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

      this.model.on('change:regl_scatterplot_options', () => {
        this.options = this.model.get('regl_scatterplot_options');
        this.optionsHandler.call(this, this.options);
      }, this);

      this.model.on('msg:custom', (event) => {
        this.customEventHandler.call(this, event);
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

      this.showTooltipDebounced = debounce(
        this.showTooltip.bind(this),
        TOOLTIP_DEBOUNCE_TIME
      );

      if (this.points.length) {
        const options = {}
        if (this.filter && this.filter.length) options.filter = this.filter;
        if (this.selection.length) options.select = this.selection;

        this.scatterplot
          .draw(this.points, options)
          .then(() => {
            if (this.annotations) {
              this.scatterplot.drawAnnotations(this.annotations);
            }
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

  customEventHandler(event) {
    if (event.type === this.eventTypes.TOOLTIP) {
      if (event.index !== this.tooltipPointIdx && event.show !== true) return;
      this.tooltipDataHandlers(event)
      if (event.show) this.showTooltip(event.index);
      return;
    }

    if (event.type === this.eventTypes.VIEW_DOWNLOAD) {
      this.viewDownload(event.transparentBackgroundColor)
      return;
    }

    if (event.type === this.eventTypes.VIEW_RESET) {
      if (!this.scatterplot) return;
      if (event.area) {
        this.scatterplot.zoomToArea(
          event.area,
          {
            transition: event.animation > 0,
            transitionDuration: event.animation,
            transitionEasing: 'quadInOut',
          }
        );
      } else {
        this.scatterplot.zoomToOrigin(
          {
            transition: event.animation > 0,
            transitionDuration: event.animation,
            transitionEasing: 'quadInOut',
          }
        );
      }
      return;
    }

    if (event.type === this.eventTypes.VIEW_SAVE) {
      this.viewSave(event.transparentBackgroundColor)
      return;
    }
  }

  getOuterDimensions() {
    let xPadding = 0;
    let yPadding = 0;

    if (this.model.get('axes')) {
      xPadding = this.getXPadding();
      yPadding = this.getYPadding();
    }

    const outerWidth = this.model.get('width') === 'auto'
      ? this.container.getBoundingClientRect().width
      : this.model.get('width') + xPadding;

    const outerHeight = this.model.get('height') + yPadding;

    this.outerWidth = outerWidth;
    this.outerHeight = outerHeight;

    return [Math.max(1, outerWidth), Math.max(1, outerHeight)];
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

    const [xLabel, yLabel] = this.model.get('axes_labels') || [];
    const xPadding = this.getXPadding();
    const yPadding = this.getYPadding();

    // Regl-Scatterplot's gl-space is always linear, hence we have to pass a
    // linear scale to regl-scatterplot.
    // In the future we might integrate this into regl-scatterplot directly
    const xScaleDomain = this.model.get('x_scale_domain');
    this.xScaleRegl = scaleLinear()
      .domain(xScaleDomain)
      .range([0, width - xPadding]);
    // This scale is used for the D3 axis
    this.xScaleAxis = getScale(this.model.get('x_scale'))
      .domain(xScaleDomain)
      .range([0, width - xPadding]);
    // This scale converts between the linear, log, or power normalized data
    // scale and the axis
    this.xScaleRegl2Axis = getScale(this.model.get('x_scale'))
      .domain(xScaleDomain)
      .range(xScaleDomain);

    const yScaleDomain = this.model.get('y_scale_domain');
    this.yScaleRegl = scaleLinear()
      .domain(yScaleDomain)
      .range([height - yPadding, 0]);
    this.yScaleAxis = getScale(this.model.get('y_scale'))
      .domain(yScaleDomain)
      .range([height - yPadding, 0]);
    this.yScaleRegl2Axis = getScale(this.model.get('y_scale'))
      .domain(yScaleDomain)
      .range(yScaleDomain);

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

    if (xLabel) {
      this.xAxisLabel = this.axesSvg.select('.x-axis-label').node()
        ? this.axesSvg.select('.x-axis-label')
        : this.axesSvg.append('text').attr('class', 'x-axis-label');

      this.xAxisLabel
        .text(xLabel)
        .attr('fill', 'currentColor')
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('x', (width - xPadding) / 2)
        .attr('y', height);
      }

    if (yLabel) {
      this.yAxisLabel = this.axesSvg.select('.y-axis-label').node()
        ? this.axesSvg.select('.y-axis-label')
        : this.axesSvg.append('text').attr('class', 'y-axis-label');

      this.yAxisLabel
        .text(yLabel)
        .attr('fill', 'currentColor')
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

    this.updateAxes(this.xScaleRegl.domain(), this.yScaleRegl.domain());
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

  createTooltipPreviewDomElements() {
    this.tooltipPreviewValue = document.createElement(
      this.model.get('tooltip_preview_type') === 'audio' ? 'audio' : 'div'
    );
    this.tooltipPreviewValueHelper = document.createElement('div');
    this.tooltipPreviewBorder = document.createElement('div');

    this.tooltipPreviewValue.appendChild(this.tooltipPreviewValueHelper);
    this.tooltipContentPreview.appendChild(this.tooltipPreviewValue);
    this.tooltipContentPreview.appendChild(this.tooltipPreviewBorder);
  }

  createTooltipPropertyDomElements(property) {
    const capitalProperty = toCapitalCase(property);
    const htmlClassProperty = toHtmlClass(property);

    this[`tooltipProperty${capitalProperty}Title`] = createElementWithClass(
      'div',
      ['title', `${htmlClassProperty}-title`]
    );

    this[`tooltipProperty${capitalProperty}Value`] = createElementWithClass(
      'div',
      ['value', `${htmlClassProperty}-value`]
    );

    this[`tooltipProperty${capitalProperty}ValueText`] = createElementWithClass(
      'div',
      ['value-text', `${htmlClassProperty}-value-text`]
    );

    this[`tooltipProperty${capitalProperty}Channel`] = createElementWithClass(
      'div',
      ['channel', `${htmlClassProperty}-channel`]
    );

    if (property in TOOLTIP_ALL_VISUAL_PROPERTIES) {
      this[`tooltipProperty${capitalProperty}ChannelName`] = createElementWithClass(
        'div',
        ['channel-name', `${htmlClassProperty}-channel-name`]
      );
      this[`tooltipProperty${capitalProperty}ChannelName`].textContent = TOOLTIP_ALL_VISUAL_PROPERTIES[property];

      this[`tooltipProperty${capitalProperty}Channel`].appendChild(
        this[`tooltipProperty${capitalProperty}ChannelName`]
      );

      this[`tooltipProperty${capitalProperty}ChannelBadge`] = createElementWithClass(
        'div',
        ['channel-badge', `${htmlClassProperty}-channel-badge`]
      );
      this[`tooltipProperty${capitalProperty}ChannelBadgeMark`] = createElementWithClass(
        'div',
        ['channel-badge-mark', `${htmlClassProperty}-channel-badge-mark`]
      );
      this[`tooltipProperty${capitalProperty}ChannelBadgeBg`] = createElementWithClass(
        'div',
        ['channel-badge-bg', `${htmlClassProperty}-channel-badge-bg`]
      );
      this[`tooltipProperty${capitalProperty}ChannelBadge`].appendChild(
        this[`tooltipProperty${capitalProperty}ChannelBadgeMark`]
      );
      this[`tooltipProperty${capitalProperty}ChannelBadge`].appendChild(
        this[`tooltipProperty${capitalProperty}ChannelBadgeBg`]
      );
      this[`tooltipProperty${capitalProperty}Channel`].appendChild(
        this[`tooltipProperty${capitalProperty}ChannelBadge`]
      );
    }

    this[`tooltipProperty${capitalProperty}Value`].appendChild(
      this[`tooltipProperty${capitalProperty}ValueText`]
    );

    this[`tooltipProperty${capitalProperty}ValueHistogram`] = createHistogram(
      TOOLTIP_HISTOGRAM_WIDTH[this.model.get('tooltip_histograms_size')] || TOOLTIP_HISTOGRAM_WIDTH.small,
      TOOLTIP_HISTOGRAM_HEIGHT[this.model.get('tooltip_histograms_size')] || TOOLTIP_HISTOGRAM_HEIGHT.small,
    );
    this[`tooltipProperty${capitalProperty}Value`].appendChild(
      this[`tooltipProperty${capitalProperty}ValueHistogram`].element
    );

    const histogram = this.model.get(`${property}_histogram`) || this.model.get('tooltip_properties_non_visual_info')[property]?.histogram;
    const scale = this.model.get(`${property}_scale`) || this.model.get('tooltip_properties_non_visual_info')[property]?.scale;

    this[`tooltipProperty${capitalProperty}ValueHistogram`].init(
      histogram,
      scale === 'categorical',
    );

    this[`tooltipProperty${capitalProperty}Title`].textContent = toTitleCase(
      this.model.get(`${property}_title`) || property || ''
    );

    this.tooltipContentProperties.appendChild(this[`tooltipProperty${capitalProperty}Channel`]);
    this.tooltipContentProperties.appendChild(this[`tooltipProperty${capitalProperty}Title`]);
    this.tooltipContentProperties.appendChild(this[`tooltipProperty${capitalProperty}Value`]);
  }

  createTooltipContentsDomElements() {
    // Remove existing DOM elements. Not the most efficient approach but it's
    // error-prone.
    this.tooltipContentPreview.replaceChildren();
    this.tooltipContentProperties.replaceChildren();

    this.createTooltipPreviewDomElements();

    const properties = new Set(this.tooltipPropertiesAll);

    for (const property of properties) {
      this.createTooltipPropertyDomElements(property);
    }

    this.styleTooltip();
  }

  createTooltipContents() {
    this.tooltipPropertiesVisual = new Set(this.model.get('tooltip_properties'));

    // Remove all but visual properties that are used for encoding data
    // properties
    for (const property of this.tooltipPropertiesVisual) {
      if (property in TOOLTIP_MANDATORY_VISUAL_PROPERTIES) continue;
      if (property in TOOLTIP_OPTIONAL_VISUAL_PROPERTIES) {
        const encoding = this.model.get(`${property}_by`);
        if (property === 'opacity' && encoding !== 'density') continue
        if (property !== 'opacity' && encoding) continue
      }
      this.tooltipPropertiesVisual.delete(property);
    }
    this.tooltipPropertiesVisual = Array.from(this.tooltipPropertiesVisual);

    this.tooltipPropertiesNonVisual = new Set(this.model.get('tooltip_properties'));
    for (const property of Object.keys(TOOLTIP_ALL_VISUAL_PROPERTIES)) {
      this.tooltipPropertiesNonVisual.delete(property);
    }
    this.tooltipPropertiesNonVisual = Array.from(this.tooltipPropertiesNonVisual);

    this.tooltipPropertiesAll = [
      ...this.tooltipPropertiesVisual,
      ...this.tooltipPropertiesNonVisual
    ];

    this.tooltipPreview = this.model.get('tooltip_preview');

    this.createTooltipContentsDomElements();
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
    this.tooltipContent.style.gridTemplateColumns = 'min-content';
    this.tooltipContent.style.borderRadius = '0.2rem';
    this.tooltip.appendChild(this.tooltipContent);

    this.tooltipContentPreview = document.createElement('div');
    this.tooltipContentPreview.style.position = 'relative';
    this.tooltipContent.appendChild(this.tooltipContentPreview);

    this.tooltipContentProperties = document.createElement('div');
    this.tooltipContentProperties.style.position = 'relative';
    this.tooltipContentProperties.style.display = 'grid';
    this.tooltipContentProperties.style.gap = '0.5em';
    this.tooltipContentProperties.style.userSelect = 'none';
    this.tooltipContentProperties.style.borderRadius = '0.2rem';
    this.tooltipContentProperties.style.padding = '0.25em';
    this.tooltipContentProperties.style.fontSize = getTooltipFontSize(this.model.get('tooltip_size'));
    this.tooltipContent.appendChild(this.tooltipContentProperties);

    this.container.appendChild(this.tooltip);

    this.createTooltipContents();

    this.enableTooltipHistograms();
    this.createTooltipContentUpdater();
  }

  positionTooltipTopCenter() {
    this.tooltipArrow.style.removeProperty('top');
    this.tooltipArrow.style.removeProperty('right');
    this.tooltipArrow.style.bottom = 0;
    this.tooltipArrow.style.left = '50%';
    this.tooltipArrow.style.transform = 'translate(-50%, calc(50% - 1px)) rotate(-45deg)';
    this.tooltipArrow.style.boxShadow = `0 0 1px rgba(0, 0, 0, ${this.tooltipOpacity}), -1px 1px 2px rgba(0, 0, 0, ${this.tooltipOpacity})`;

    this.moveTooltip = (x, y) => {
      this.tooltip.style.transform = `translate(calc(${x}px - 50%), calc(${y}px - ${TOOLTIP_OFFSET_REM}rem - 100%))`;
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
      this.tooltip.style.transform = `translate(calc(${x}px - 100% + ${TOOLTIP_OFFSET_REM}rem), calc(${y}px - ${TOOLTIP_OFFSET_REM}rem - 100%))`;
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
      this.tooltip.style.transform = `translate(calc(${x}px - ${TOOLTIP_OFFSET_REM}rem), calc(${y}px - ${TOOLTIP_OFFSET_REM}rem - 100%))`;
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
      this.tooltip.style.transform = `translate(calc(${x}px - 50%), calc(${y}px + ${TOOLTIP_OFFSET_REM}rem))`;
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
      this.tooltip.style.transform = `translate(calc(${x}px - 100% + ${TOOLTIP_OFFSET_REM}rem), calc(${y}px + ${TOOLTIP_OFFSET_REM}rem))`;
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
      this.tooltip.style.transform = `translate(calc(${x}px - ${TOOLTIP_OFFSET_REM}rem), calc(${y}px + ${TOOLTIP_OFFSET_REM}rem))`;
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
      this.tooltip.style.transform = `translate(calc(${x}px - ${TOOLTIP_OFFSET_REM}rem - 100%), calc(${y}px - 50%))`;
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
      this.tooltip.style.transform = `translate(calc(${x}px + ${TOOLTIP_OFFSET_REM}rem), calc(${y}px - 50%))`;
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
    const { width, height } = this.tooltip.getBoundingClientRect();
    const xCutoff = width / 2;
    const yCutoff = height;
    const tooltipOffset = remToPx(TOOLTIP_OFFSET_REM);

    if (x < xCutoff + tooltipOffset) {
      if (y < yCutoff + tooltipOffset) return 'bottom-right';
      if (y > this.outerHeight - yCutoff - tooltipOffset) return 'top-right';
      return 'right-center';
    }

    if (x > this.outerWidth - xCutoff - tooltipOffset) {
      if (y < yCutoff + tooltipOffset) return 'bottom-left';
      if (y > this.outerHeight - yCutoff - tooltipOffset) return 'top-left';
      return 'left-center';
    }

    if (y < yCutoff + tooltipOffset) return 'bottom-center';

    return 'top-center';
  }

  enableTooltipHistograms() {
    const display = this.model.get('tooltip_histograms') ? 'block' : 'none';
    const histograms = this.tooltipContentProperties.querySelectorAll('.histogram');

    for (const histogram of histograms) {
      histogram.style.display = display;
    }
  }

  styleTooltip() {
    if (!this.tooltip) this.createTooltip();
    const color = this.model.get('tooltip_color').map((c) => Math.round(c * 255));

    const isDark = color[0] <= 127;
    const contrast = isDark ? 255 : 0;
    const fg = `rgb(${contrast}, ${contrast}, ${contrast})`;
    const bg = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

    for (const property of this.tooltipPropertiesAll) {
      this[`tooltipProperty${toCapitalCase(property)}ValueHistogram`]?.style(
        fg,
        `rgb(${contrast}, ${contrast}, ${contrast}, 0.2)`
      );
    }

    if (this.tooltipPropertiesVisual.length) {
      this.tooltipContentProperties.style.gridTemplateColumns = 'max-content max-content max-content';
    } else {
      // Let's hide the channel column since no properties is visually encoded
      this.tooltipContentProperties.style.gridTemplateColumns = 'max-content max-content';
      for (const channel of this.tooltipContentProperties.querySelectorAll('.channel')) {
        channel.style.display = 'none';
      }
    }

    this.tooltipOpacity = isDark ? 0.33 : 0.2;

    this.tooltip.style.opacity = 0;
    this.tooltip.style.color = fg;
    this.tooltip.style.backgroundColor = bg;
    this.tooltip.style.boxShadow = `0 0 1px rgba(0, 0, 0, ${this.tooltipOpacity}), 0 1px 2px rgba(0, 0, 0, ${this.tooltipOpacity})`;
    this.tooltipArrow.style.backgroundColor = bg;
    this.tooltipContent.style.backgroundColor = bg;

    if (this.model.get('tooltip_preview')) {
      const previewType = this.model.get('tooltip_preview_type');

      if (previewType === 'text') {
        const lines = this.model.get('tooltip_preview_text_lines');

        this.tooltipPreviewValue.style.position = 'relative';
        this.tooltipPreviewValue.style.width = '100%';
        this.tooltipPreviewValueHelper.style.margin = '0.25em';
        this.tooltipPreviewBorder.style.height = '1px';
        this.tooltipPreviewBorder.style.marginBottom = '0.25em';
        this.tooltipPreviewBorder.style.backgroundColor = `rgb(${contrast}, ${contrast}, ${contrast}, 0.2)`;

        if (lines > 0) {
          this.tooltipPreviewValueHelper.style.display = '-webkit-box';
          this.tooltipPreviewValueHelper.style.webkitLineClamp = lines;
          this.tooltipPreviewValueHelper.style.webkitBoxOrient = 'vertical';
          this.tooltipPreviewValueHelper.style.overflow = 'hidden';
        }
      } else if (previewType === 'image') {
        const backgroundColor = this.model.get('tooltip_preview_image_background_color');
        const position = this.model.get('tooltip_preview_image_position');
        const size = this.model.get('tooltip_preview_image_size');

        this.tooltipPreviewValue.style.position = 'relative';
        this.tooltipPreviewValue.style.backgroundColor =
          backgroundColor === 'auto' ? bg : backgroundColor;
        this.tooltipPreviewValue.style.backgroundRepeat = 'no-repeat';
        this.tooltipPreviewValue.style.backgroundPosition = position;
        this.tooltipPreviewValue.style.backgroundSize = size;
        this.tooltipPreviewValue.style.borderRadius = '0.2rem 0.2rem 0 0';

        this.tooltipPreviewValueHelper.style.paddingTop = '6em';

        this.tooltipPreviewBorder.style.height = '1px';
        this.tooltipPreviewBorder.style.marginBottom = '0.25em';
        this.tooltipPreviewBorder.style.backgroundColor = `rgb(${contrast}, ${contrast}, ${contrast}, 0.2)`;
      } else if (previewType === 'audio') {
        const length = this.model.get('tooltip_preview_audio_length');
        const loop = this.model.get('tooltip_preview_audio_loop');
        const controls = this.model.get('tooltip_preview_audio_controls');
        this.tooltipPreviewValue.controls = controls;
        this.tooltipPreviewValue.autoplay = true;
        this.tooltipPreviewValue.loop = loop;

        if (length) {
          this.tooltipPreviewValue.addEventListener("timeupdate", () => {
            if (this.tooltipPreviewValue.currentTime > length) {
              this.tooltipPreviewValue.pause();
              if (loop) {
                this.tooltipPreviewValue.currentTime = 0;
                this.tooltipPreviewValue.play();
              }
            }
          });
        }
      }
    }

    const channelBadges = this.tooltipContentProperties.querySelectorAll('.channel-badge');

    for (const channelBadge of channelBadges) {
      channelBadge.style.position = 'relative';
      channelBadge.style.fontSize = '0.75em';
      channelBadge.style.width = '1em';
      channelBadge.style.height = '1em';
      channelBadge.style.marginRight = '0.125rem';
    }

    const channelBadgeBgs = this.tooltipContentProperties.querySelectorAll('.channel-badge-bg');

    for (const channelBadgeBg of channelBadgeBgs) {
      channelBadgeBg.style.position = 'absolute';
      channelBadgeBg.style.zIndex = 2;
      channelBadgeBg.style.top = 0;
      channelBadgeBg.style.left = 0;
      channelBadgeBg.style.width = '1em';
      channelBadgeBg.style.height = '1em';
    }

    if (this.tooltipPropertyXChannelBadgeBg) {
      this.tooltipPropertyXChannelBadgeBg.style.zIndex = 0;
      this.tooltipPropertyXChannelBadgeBg.style.top = '50%';
      this.tooltipPropertyXChannelBadgeBg.style.transform = 'translate(0, -50%)';
      this.tooltipPropertyXChannelBadgeBg.style.height = '2px';
      this.tooltipPropertyXChannelBadgeBg.style.background = `rgba(${contrast}, ${contrast}, ${contrast}, 0.2)`;

      this.tooltipPropertyXChannelBadgeMark.style.position = 'absolute';
      this.tooltipPropertyXChannelBadgeMark.style.zIndex = 1;
      this.tooltipPropertyXChannelBadgeMark.style.top = '50%';
      this.tooltipPropertyXChannelBadgeMark.style.width = '2px';
      this.tooltipPropertyXChannelBadgeMark.style.height = '6px';
      this.tooltipPropertyXChannelBadgeMark.style.background = `rgb(${contrast}, ${contrast}, ${contrast})`;
    }

    if (this.tooltipPropertyYChannelBadgeBg) {
      this.tooltipPropertyYChannelBadgeBg.style.zIndex = 0;
      this.tooltipPropertyYChannelBadgeBg.style.left = '50%';
      this.tooltipPropertyYChannelBadgeBg.style.transform = 'translate(-50%, 0)';
      this.tooltipPropertyYChannelBadgeBg.style.width = '2px';
      this.tooltipPropertyYChannelBadgeBg.style.background = `rgba(${contrast}, ${contrast}, ${contrast}, 0.2)`;

      this.tooltipPropertyYChannelBadgeMark.style.position = 'absolute';
      this.tooltipPropertyYChannelBadgeMark.style.zIndex = 1;
      this.tooltipPropertyYChannelBadgeMark.style.left = '50%';
      this.tooltipPropertyYChannelBadgeMark.style.width = '6px';
      this.tooltipPropertyYChannelBadgeMark.style.height = '2px';
      this.tooltipPropertyYChannelBadgeMark.style.background = `rgb(${contrast}, ${contrast}, ${contrast})`;
    }

    [
      this.tooltipPropertyOpacityChannelBadgeBg,
      this.tooltipPropertySizeChannelBadgeBg,
    ].filter((x) => x).forEach((element) => {
      element.style.borderRadius = '1em';
      element.style.boxShadow = `inset 0 0 0 1px rgba(${contrast}, ${contrast}, ${contrast}, 0.2)`;
    });

    [
      this.tooltipPropertyColorChannelBadgeMark,
      this.tooltipPropertyOpacityChannelBadgeMark,
      this.tooltipPropertySizeChannelBadgeMark,
    ].filter((x) => x).forEach((element) => {
      element.style.width = '1em';
      element.style.height = '1em';
      element.style.borderRadius = '1em';
    });

    [
      this.tooltipPropertyOpacityChannelBadgeMark,
      this.tooltipPropertySizeChannelBadgeMark,
    ].filter((x) => x).forEach((element) => {
      element.style.background = `rgb(${contrast}, ${contrast}, ${contrast})`;
    });

    const channelBg = `rgba(${contrast}, ${contrast}, ${contrast}, 0.075)`;
    const channelColor = `rgba(${contrast}, ${contrast}, ${contrast}, 0.5)`;

    [
      this.tooltipPropertyXChannel,
      this.tooltipPropertyYChannel,
      this.tooltipPropertyColorChannel,
      this.tooltipPropertyOpacityChannel,
      this.tooltipPropertySizeChannel,
    ].filter((x) => x).forEach((el) => {
      el.style.display = 'flex';
      el.style.gap = '0 0.33em';
      el.style.justifyContent = 'center';
      el.style.alignItems = 'center';
      el.style.padding = '0 0.25em';
      el.style.borderRadius = '0.25rem';
      el.style.color = channelColor;
      el.style.height = '1.3em';
      el.style.fontWeight = 'bold';
      el.style.textTransform = 'uppercase';
      el.style.background = channelBg;
    });

    [
      this.tooltipPropertyXChannelName,
      this.tooltipPropertyYChannelName,
      this.tooltipPropertyColorChannelName,
      this.tooltipPropertyOpacityChannelName,
      this.tooltipPropertySizeChannelName,
    ].filter((x) => x).forEach((el) => {
      el.style.display = 'flex';
      el.style.flexGrow = '1';
      el.style.justifyContent = 'center';
      el.style.alignItems = 'center';
      el.style.fontSize = '0.75em';
      el.style.lineHeight = '1em';
    });

    const values = this.tooltipContentProperties.querySelectorAll('.value');

    for (const value of values) {
      value.style.display = 'flex';
      value.style.gap = '0 0.25em';
      value.style.fontWeight = 'bold';
      value.style.alignItems = 'top';
    }

    const valueTexts = this.tooltipContentProperties.querySelectorAll('.value-text');

    for (const valueText of valueTexts) {
      valueText.style.flexGrow = '1';
      valueText.style.maxWidth = '12em';
      valueText.style.display = '-webkit-box';
      valueText.style.webkitLineClamp = 3;
      valueText.style.webkitBoxOrient = 'vertical';
      valueText.style.overflow = 'hidden';
    }
  }

  getPoint(i) {
    return this.scatterplot.get('points')[i];
  }

  getPoints() {
    return this.scatterplot.get('points');
  }

  createXGetter() {
    if (!this.xScale) this.createXScale();

    this.getXBin = createNumericalBinGetter(
      this.model.get('x_histogram'),
      this.model.get('x_domain'),
    );

    const toRelValue = scaleLinear()
      .domain(this.model.get('x_domain'))
      .range([0, 1]);

    this.getX = (i) => {
      const ndc = this.getPoint(i)[0];
      const value = this.xScale.invert(ndc);
      return [
        toRelValue(value),
        this.xFormat(value),
        this.getXBin(value)
      ];
    }
  }

  createYGetter() {
    if (!this.yScale) this.createYScale();

    this.getYBin = createNumericalBinGetter(
      this.model.get('y_histogram'),
      this.model.get('y_domain'),
    );

    const toRelValue = scaleLinear()
      .domain(this.model.get('y_domain'))
      .range([0, 1]);

    this.getY = (i) => {
      const ndc = this.getPoint(i)[1];
      const value = this.yScale.invert(ndc);
      return [
        toRelValue(value),
        this.yFormat(value),
        this.getYBin(value),
      ];
    }
  }

  createColorGetter() {
    if (!this.colorScale) this.createColorScale();
    if (!this.colorScale) {
      this.getColor = () => ['#808080', 'Unknown', 0];
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
          value
        ]
      }
    } else {
      const numColors = colors.length;

      this.getColorBin = createNumericalBinGetter(
        this.model.get('color_histogram'),
        this.model.get('color_domain'),
      );

      this.getColor = (i) => {
        const normalizedValue = this.getPoint(i)[dim];
        const colorIdx = Math.min(numColors - 1, Math.floor(numColors * normalizedValue));
        const value = this.colorScale.invert(normalizedValue);
        return [
          colors[colorIdx] || '#808080',
          this.colorFormat(value),
          this.getColorBin(value),
        ]
      }
    }
  }

  createOpacityGetter() {
    if (!this.opacityScale) this.createOpacityScale();
    if (!this.opacityScale) {
      this.getOpacity = () => [0.5, 'Unknown', 0];
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
          value,
        ]
      }
    } else {
      const numOpacities = opacities.length;

      this.getOpacityBin = createNumericalBinGetter(
        this.model.get('opacity_histogram'),
        this.model.get('opacity_domain'),
      );

      this.getOpacity = (i) => {
        const normalizedValue = this.getPoint(i)[dim];
        const idx = Math.min(numOpacities - 1, Math.floor(numOpacities * normalizedValue));
        const value = this.opacityScale.invert(normalizedValue);
        return [
          opacities[idx] || 0.5,
          this.opacityFormat(value),
          this.getOpacityBin(value),
        ]
      }
    }
  }

  createSizeGetter() {
    if (!this.sizeScale) this.createSizeScale();
    if (!this.sizeScale) {
      this.getSize = () => [0.5, 'Unknown', 0];
      return;
    }

    const dim = this.model.get('size_by') === 'valueA' ? 2 : 3;
    const sizes = this.model.get('size');
    const sizesMin = Array.isArray(sizes) ? min(sizes) : sizes;
    const sizesMax = Array.isArray(sizes) ? max(sizes) : sizes;
    const sizesExtent = (sizesMax - sizesMin) || 1;

    if (this.model.get('size_scale') === 'categorical') {
      this.getSize = (i) => {
        const value = this.getPoint(i)[dim];
        return [
          sizes[value] !== undefined
            ? Math.max(0.1, (sizes[value] - sizesMin) / sizesExtent)
            : '#808080',
          this.sizeFormat(this.sizeScale.invert(value)),
          value,
        ]
      }
    } else {
      const numSizes = sizes.length;

      this.getSizeBin = createNumericalBinGetter(
        this.model.get('size_histogram'),
        this.model.get('size_domain'),
      );

      this.getSize = (i) => {
        const normalizedValue = this.getPoint(i)[dim];
        const idx = Math.min(numSizes - 1, Math.floor(numSizes * normalizedValue));
        const value = this.sizeScale.invert(normalizedValue);
        return [
          Math.max(0.1, (sizes[idx] - sizesMin) / sizesExtent),
          this.sizeFormat(value),
          this.getSizeBin(value),
        ]
      }
    }
  }

  createTooltipContentUpdater() {
    const visualUpdaters = this.tooltipPropertiesVisual
      .map((property) => {
        const propertyTitle = toCapitalCase(property);
        const get = (pointIdx) => this[`get${propertyTitle}`](pointIdx);

        const textElement = this[`tooltipProperty${propertyTitle}ValueText`];
        const badgeElement = this[`tooltipProperty${propertyTitle}ChannelBadgeMark`];
        const histogram = this[`tooltipProperty${propertyTitle}ValueHistogram`];

        if (property === 'x') {
          return (pointIdx) => {
            const [x, text, histogramKey] = get(pointIdx);
            badgeElement.style.transform = `translate(calc(${x}em - 50%), -50%)`;
            textElement.textContent = text;
            if (this.model.get('tooltip_histograms')) histogram.draw(histogramKey);
          }
        }

        if (property === 'y') {
          return (pointIdx) => {
            const [y, text, histogramKey] = get(pointIdx);
            badgeElement.style.transform = `translate(-50%, calc(${1 - y}em - 50%))`;
            textElement.textContent = text;
            if (this.model.get('tooltip_histograms')) histogram.draw(histogramKey);
          }
        }

        if (property === 'color') {
          return (pointIdx) => {
            const [color, text, histogramKey] = get(pointIdx);
            badgeElement.style.background = color;
            textElement.textContent = text;
            if (this.model.get('tooltip_histograms')) histogram.draw(histogramKey);
          }
        }

        if (property === 'opacity') {
          return (pointIdx) => {
            const [opacity, text, histogramKey] = get(pointIdx);
            badgeElement.style.opacity = opacity;
            textElement.textContent = text;
            if (this.model.get('tooltip_histograms')) histogram.draw(histogramKey);
          }
        }

        if (property === 'size') {
          return (pointIdx) => {
            const [scale, text, histogramKey] = get(pointIdx);
            badgeElement.style.transform = `scale(${scale})`;
            textElement.textContent = text;
            if (this.model.get('tooltip_histograms')) histogram.draw(histogramKey);
          }
        }

        return (pointIdx) => textElement.textContent = get(pointIdx);
      });

    const nonVisualInfo = this.model.get('tooltip_properties_non_visual_info');
    const nonVisualData = this.tooltipPropertiesNonVisual.map((property) => {
      const propertyTitle = toCapitalCase(property);

      const textElement = this[`tooltipProperty${propertyTitle}ValueText`];
      const histogram = this[`tooltipProperty${propertyTitle}ValueHistogram`];

      const info = nonVisualInfo[property];

      return {
        property,
        textElement,
        histogram,
        getHistogramKey: info.scale === 'categorical'
          ? (v) => info.domain[v]
          : createNumericalBinGetter(info.histogram, info.range || info.domain),
        format: info.scale === 'categorical'
          ? (s) => s
          : format(getD3FormatSpecifier(info.domain))
      }
    });

    const previewType = this.model.get('tooltip_preview_type');

    let previewUpdater;

    if (previewType === 'text') {
      previewUpdater = (text) => {
        this.tooltipPreviewValueHelper.textContent = text;
      }
    }

    if (previewType === 'image') {
      previewUpdater = (imageUrl) => {
        this.tooltipPreviewValue.style.backgroundImage = `url(${imageUrl})`;
      }
    }

    if (previewType === 'audio') {
      previewUpdater = (audioSrc) => {
        this.tooltipPreviewValue.src = audioSrc;
        this.tooltipPreviewValue.currentTime = 0;
      }
    }

    this.tooltipDataHandlers = (event) => {
      for (const d of nonVisualData) {
        if (!(d.property in event.properties)) continue;
        const value = event.properties[d.property];
        d.textElement.textContent = d.format(value);
        d.histogram.draw(d.getHistogramKey(value));
      }
      if (event.preview) previewUpdater(event.preview);
    }

    this.tooltipContentsUpdater = (pointIdx) => {
      this.model.send({
        type: this.eventTypes.TOOLTIP,
        index: pointIdx,
        properties: this.tooltipPropertiesNonVisual,
        preview: this.tooltipPreview,
      });
      visualUpdaters.forEach((updater) => { updater(pointIdx); });
    }
  }

  showTooltip(pointIdx) {
    this.tooltipPointIdx = pointIdx;

    const [x, y] = this.scatterplot.getScreenPosition(pointIdx);

    const newTooltipPosition = this.getTooltipPosition(x, y);
    if (newTooltipPosition !== this.tooltipPosition) {
      this.tooltipPosition = newTooltipPosition;
      this.positionTooltip(this.tooltipPosition);
    }

    this.moveTooltip(x, y);
    this.tooltipContentsUpdater(pointIdx);
    this.tooltip.style.opacity = 1;
  }

  hideTooltip() {
    this.tooltipPointIdx = undefined;
    this.tooltip.style.opacity = 0;
    if (this.tooltipPreviewValue.nodeName === 'AUDIO') {
      this.tooltipPreviewValue.pause();
      this.tooltipPreviewValue.currentTime = 0;
    }
  }

  tooltipEnableHandler(tooltip) {
    if (!tooltip) this.hideTooltip;
  }

  tooltipSizeHandler(newSize) {
    this.tooltipContent.style.fontSize = getTooltipFontSize(newSize);
    for (const property of this.tooltipPropertiesAll) {
      this[`tooltipProperty${toCapitalCase(property)}ValueHistogram`].resize();
    }
  }

  tooltipColorHandler() {
    this.styleTooltip();
  }

  refreshTooltip() {
    this.createTooltipContents();
    this.createTooltipContentUpdater();
  }

  tooltipPropertiesHandler() {
    this.refreshTooltip();
  }

  tooltipPropertiesNonVisualInfoHandler() {
    this.refreshTooltip();
  }

  tooltipHistogramsSizeHandler() {
    this.refreshTooltip();
  }

  tooltipHistogramsRangesHandler() {
    this.createXGetter();
    this.createYGetter();
    this.createColorGetter();
    this.createOpacityGetter();
    this.createSizeGetter();
    this.refreshTooltip();
  }

  tooltipHistogramsHandler() {
    this.enableTooltipHistograms();
  }

  updateLegendWrapperPosition() {
    if (!this.legendWrapper) return;

    this.legendWrapper.style.top = 0;
    this.legendWrapper.style.bottom = this.getYPadding() + 'px';
    this.legendWrapper.style.left = 0;
    this.legendWrapper.style.right = this.getXPadding() + 'px';
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
      xPadding = this.getXPadding();
      yPadding = this.getYPadding();
    }

    this.container.style.width = width === 'auto'
      ? '100%'
      : (width + xPadding) + 'px';
    this.container.style.height = (height + yPadding) + 'px';

    window.requestAnimationFrame(() => { this.resizeHandlerBound(); });
  }

  resizeHandler() {
    if (!this.model.get('axes')) return;

    const [width, height] = this.getOuterDimensions();

    const [xLabel, yLabel] = this.model.get('axes_labels') || [];
    const xPadding = this.getXPadding();
    const yPadding = this.getYPadding();

    const xScaleDomain = this.scatterplot.get('xScale').domain();
    const yScaleDomain = this.scatterplot.get('yScale').domain();
    this.xScaleAxis.domain(xScaleDomain.map(this.xScaleRegl2Axis.invert));
    this.yScaleAxis.domain(yScaleDomain.map(this.yScaleRegl2Axis.invert));

    this.xAxisContainer.call(this.xAxis.scale(this.xScaleAxis));
    this.yAxisContainer.call(this.yAxis.scale(this.yScaleAxis));

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
      this.axesSvg.selectAll('line')
        .attr('stroke-opacity', 0.2)
        .attr('stroke-dasharray', 2);
    }

    if (xLabel) {
      this.xAxisLabel.attr('x', (width - xPadding) / 2).attr('y', height - 4);
    }
    if (yLabel) {
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
    this.showTooltipDebounced.cancel();
    this.scatterplot.destroy();
  }

  // Helper
  colorCanvas() {
    if (Array.isArray(this.backgroundColor)) {
      const rgbStr =
        this.backgroundColor.slice(0, 3).map((x) => x * 255).join(',');

      const colorStr = this.backgroundColor.length === 4
        ? `rgba(${rgbStr}, ${this.backgroundColor[3]})`
        : `rgb(${rgbStr})`;

      this.container.style.backgroundColor = colorStr;
    } else {
      this.container.style.backgroundColor = this.backgroundColor;
    }
  }

  annotationsHandler(annotations) {
    this.scatterplot.drawAnnotations(annotations || []);
  }

  // Event handlers for JS-triggered events
  pointoverHandler(pointIndex) {
    this.hoveringChangedByJs = true;
    if (this.model.get('tooltip_enable')) this.showTooltipDebounced(pointIndex);
    this.model.set('hovering', pointIndex);
    this.model.save_changes();
  }

  pointoutHandler() {
    this.hoveringChangedByJs = true;
    this.showTooltipDebounced.cancel();
    this.hideTooltip();
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
      if (this.model.get('axes') && event.xScaleDomain && event.yScaleDomain) {
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
          xScaleDomain: event.xScale?.domain(),
          yScaleDomain: event.yScale?.domain(),
        },
        { async: true }
      );
    }
    if (this.model.get('axes')) {
      this.updateAxes(event.xScale.domain(), event.yScale.domain());
    }
  }

  createXScale() {
    const domain = this.model.get('x_scale_domain');
    const scale = this.model.get('x_scale');
    this.xScale = getScale(scale)
      .domain(domain)
      .range([-1, 1]);
    this.xFormat =
      scale === 'time'
        ? createTimeFormat(
            this.points,
            (point) => Math.floor(this.xScale.invert(point[0]).getTime() / 1000)
          )
        : format(getD3FormatSpecifier(domain));
  }

  createYScale() {
    const domain = this.model.get('y_scale_domain');
    const scale = this.model.get('y_scale');
    this.yScale = getScale(scale)
      .domain(domain)
      .range([-1, 1]);
    this.yFormat =
      scale === 'time'
        ? createTimeFormat(this.points, (p) => this.yScale.invert(p[1]))
        : format(getD3FormatSpecifier(domain));
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

  xScaleDomainHandler() {
    this.createXScale();
    if (this.model.get('axes')) this.createAxes();
  }

  yScaleDomainHandler() {
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

  xHistogramHandler() {
    this.tooltipPropertyXValueHistogram?.init(this.model.get('x_histogram'));
    this.createXGetter();
  }

  xHistogramRangeHandler() {
    this.createXGetter();
  }

  yHistogramHandler() {
    this.tooltipPropertyYValueHistogram?.init(this.model.get('y_histogram'));
    this.createYGetter();
  }

  yHistogramRangeHandler() {
    this.createYGetter();
  }

  colorHistogramHandler() {
    this.tooltipPropertyColorValueHistogram?.init(
      this.model.get('color_histogram'),
      this.model.get('color_scale') === 'categorical',
    );
    this.createColorGetter();
  }

  colorHistogramRangeHandler() {
    this.createColorGetter();
  }

  opacityHistogramHandler() {
    this.tooltipPropertyOpacityValueHistogram?.init(
      this.model.get('opacity_histogram'),
      this.model.get('opacity_scale') === 'categorical',
    );
    this.createOpacityGetter();
  }

  opacityHistogramRangeHandler() {
    this.createOpacityGetter();
  }

  sizeHistogramHandler() {
    this.tooltipPropertySizeValueHistogram?.init(
      this.model.get('size_histogram'),
      this.model.get('size_scale') === 'categorical',
    );
    this.createSizeGetter();
  }

  sizeHistogramRangeHandler() {
    this.createSizeGetter();
  }

  // Event handlers for Python-triggered events
  pointsHandler(newPoints) {
    if (newPoints.length === this.scatterplot.get('points').length) {
      // We assume point correspondence
      this.scatterplot.draw(newPoints, {
        transition: this.model.get('transition_points'),
        transitionDuration: this.model.get('transition_points_duration'),
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
    if (this.tooltipPropertyXTitle) {
      this.tooltipPropertyXTitle.textContent = toTitleCase(newTitle || '');
    }
  }

  yTitleHandler(newTitle) {
    if (this.tooltipPropertyYTitle) {
      this.tooltipPropertyYTitle.textContent = toTitleCase(newTitle || '');
    }
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
    const currValue = this.scatterplot.get('colorBy');
    this.createColorScale();
    this.createColorGetter();
    this.withPropertyChangeHandler('colorBy', newValue);
    if (!currValue && newValue) {
      // We need to reapply the point color due to some internal
      // regl-scatterplot logic which uses a different active point color when
      // the point color is changed and colorBy is undefined
      this.withPropertyChangeHandler('pointColor', this.model.get('color'));
    }
  }

  colorTitleHandler(newTitle) {
    if (this.tooltipPropertyColorTitle) {
      this.tooltipPropertyColorTitle.textContent = toCapitalCase(newTitle || '');
    }
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
    this.createOpacityScale();
    this.createOpacityGetter();
    this.withPropertyChangeHandler('opacityBy', newValue);
  }

  opacityTitleHandler(newTitle) {
    if (this.tooltipPropertyOpacityTitle) {
      this.tooltipPropertyOpacityTitle.textContent = toCapitalCase(newTitle || '');
    }
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
    if (this.tooltipPropertySizeTitle) {
      this.tooltipPropertySizeTitle.textContent = toCapitalCase(newTitle || '');
    }
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
    const currValue = this.scatterplot.get('pointConnectionColorBy');
    this.withPropertyChangeHandler('pointConnectionColorBy', newValue);

    if (currValue === 'segment' || newValue === 'segment') {
      // We need to fix this in regl-scatterplot and regl-line but changing from
      // or to color the point connections by segment, requires recreating the
      // point connections as the line's color indices change.
      // As a workaround, redrawing the points triggers the recreation of the
      // point connections.
      this.scatterplot.draw(this.scatterplot.get('points'));
    }
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
    this.scatterplot.get('camera').config({ isRotate: newValue === 'rotate' });
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

  viewDownload(transparentBackgroundColor) {
    const image = this.scatterplot.export();
    const finalImage = transparentBackgroundColor
      ? image
      : addBackgroundColor(image, this.backgroundColor);
    imageDataToCanvas(finalImage).toBlob((blob) => {
      downloadBlob(blob, 'scatter.png');
    });
  }

  viewSave(transparentBackgroundColor) {
    const image = this.scatterplot.export();
    const finalImage = transparentBackgroundColor
      ? image
      : addBackgroundColor(image, this.backgroundColor);
    this.model.set('view_data', finalImage);
    this.model.save_changes();
  }

  optionsHandler(newOptions) {
    this.scatterplot.set(newOptions);
  }

  withPropertyChangeHandler(property, changedValue) {
    const p = {};
    p[property] = changedValue;
    this.scatterplot.set(p);
  }

  getXPadding() {
    const yLabel = this.model.get('axes_labels')?.[1];
    return yLabel ? AXES_PADDING_X_WITH_LABEL : AXES_PADDING_X;
  }

  getYPadding() {
    const xLabel = this.model.get('axes_labels')?.[0];
    return xLabel ? AXES_PADDING_Y_WITH_LABEL : AXES_PADDING_Y;
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
    send: model.send.bind(model),
  }
}

async function render({ model, el }) {
  const view = new JupyterScatterView({
    el: el,
    model: modelWithSerializers(model, {
      points: Numpy2D('float32'),
      selection: Numpy1D('uint32'),
      filter: Numpy1D('uint32'),
      view_data: NumpyImage(),
      zoom_to: Numpy1D('uint32'),
      annotations: Annotations(),
    }),
  });
  view.render();
  return () => view.destroy();
}

export default { render };
