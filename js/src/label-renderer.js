const ASINH_1 = Math.asinh(1);
const LINE_HEIGHT = 1.2;
const OUT_OF_VIEW_PADDING = 0.05;
const SHADOW_COLOR = 'black';

const DEFAULT_TEXT_ALIGN = 'center';
const DEFAULT_FONT_SHADOW_COLOR = 'white';

const asinhZoomScale = (zoomScale) => {
  if (zoomScale > 1) {
    return Math.asinh(zoomScale) / ASINH_1;
  }
  return 1;
};

/**
 * Render labels
 * @param {HTMLCanvasElement} canvas Drawing canvas
 */
export const createLabelRenderer = (canvas) => {
  const ctx = canvas.getContext('2d');

  /**
   * Draw text with shadow and even opacity
   * @param{string} text The text label
   * @param{number} x X position
   * @param{number} y Y position
   * @param{string} fillColor Text color
   * @param{number} opacity Opacity at which to draw the outlined text
   * @param{string} shadowColor Shadow color
   */
  function drawTextLineWithShadow(
    text,
    x,
    y,
    fillColor,
    opacity,
    shadowColor = SHADOW_COLOR,
  ) {
    ctx.globalAlpha = opacity;
    ctx.shadowColor = shadowColor;
    ctx.shadowBlur = ctx.lineWidth;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.fillStyle = fillColor;
    ctx.fillText(text, x, y);
  }

  /**
   * Draw multi-line text with outline and even opacity, supporting line breaks
   * @param{string} text The text label
   * @param{number} x X position
   * @param{number} y Y position
   * @param{string} fillColor Text color
   * @param{string} strokeColor Outline color
   * @param{number} opacity Opacity at which to draw the outlined text
   */
  function drawMultiLineOutlinedText(
    text,
    x,
    y,
    fillColor,
    opacity,
    shadowColor,
  ) {
    // Split text by line breaks
    const lines = text.split('\n');

    // Get the font metrics to calculate line height
    const fontMetrics = ctx.measureText('Mg');
    // Approximate line height based on font metrics
    const lineHeight =
      (fontMetrics.actualBoundingBoxAscent +
        fontMetrics.actualBoundingBoxDescent) *
      LINE_HEIGHT;

    // Calculate starting y position based on number of lines and text alignment
    let startY = y;
    if (ctx.textBaseline === 'middle') {
      startY = y - (lineHeight * (lines.length - 1)) / 2;
    } else if (ctx.textBaseline === 'bottom') {
      startY = y - lineHeight * (lines.length - 1);
    }

    // Draw each line separately
    for (let i = 0; i < lines.length; i++) {
      const lineY = startY + i * lineHeight;
      drawTextLineWithShadow(
        lines[i],
        x,
        lineY,
        fillColor,
        opacity,
        shadowColor,
      );
    }
  }

  /**
   * Draw text, handling multi-line text if needed
   * @param{string} text The text label
   * @param{number} x X position
   * @param{number} y Y position
   * @param{string} fillColor Text color
   * @param{number} opacity Opacity at which to draw the outlined text
   * @param{string} shadowColor Shadow color
   */
  function drawText(text, x, y, fillColor, opacity, shadowColor) {
    if (text.includes('\n')) {
      drawMultiLineOutlinedText(text, x, y, fillColor, opacity, shadowColor);
    } else {
      drawTextLineWithShadow(text, x, y, fillColor, opacity, shadowColor);
    }
  }

  /**
   * Set text alignment
   * @param {'top'|'top-left'|'top-right'|'bottom'|'bottom-left'|'bottom-right'|'left'|'right'|'center'} textAlign The text alignment
   */
  const setTextAlignment = (textAlign) => {
    switch (textAlign) {
      case 'top': {
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        break;
      }
      case 'top-left': {
        ctx.textAlign = 'right';
        ctx.textBaseline = 'bottom';
        break;
      }
      case 'top-right': {
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        break;
      }
      case 'bottom': {
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        break;
      }
      case 'bottom-left': {
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        break;
      }
      case 'bottom-right': {
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        break;
      }
      case 'left': {
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        break;
      }
      case 'right': {
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        break;
      }
      default: {
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        break;
      }
    }
  };

  const getVisibibleDomains = (xScale, yScale) => {
    const [minDataX, maxDataX] = xScale.domain();
    const [minDataY, maxDataY] = yScale.domain();
    const outOfViewXPadding = (maxDataX - minDataX) * OUT_OF_VIEW_PADDING;
    const outOfViewYPadding = (maxDataY - minDataY) * OUT_OF_VIEW_PADDING;

    return [
      minDataX - outOfViewXPadding,
      maxDataX + outOfViewXPadding,
      minDataY - outOfViewYPadding,
      maxDataY + outOfViewYPadding,
    ];
  };

  /**
   * Clear labels
   */
  const clear = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  /**
   * Render labels
   * @param {Table} labels Labels
   * @param {number} zoom Zoom factor
   * @param {xScale} xScale x scale function
   * @param {yScale} yScale y scale function
   * @param {Object} options Rendering options
   * @param {string} options.color Text color
   * @param {string} options.outlineColor Text outline color
   * @param {string} options.scaleFunction Label scale function
   * @param {number} options.backgroundLuminance Background luminance
   */
  const render = (labels, zoom, xScale, yScale, options) => {
    const t0 = performance.now();
    clear();

    const dpr = window.devicePixelRatio;
    const [minDataX, maxDataX, minDataY, maxDataY] = getVisibibleDomains(
      xScale,
      yScale,
    );

    ctx.lineWidth = 4 * dpr;

    setTextAlignment(options.align || DEFAULT_TEXT_ALIGN);

    const fontScale =
      options.scaleFunction === 'constant' ? 1 : asinhZoomScale(zoom);

    const [xOffset, yOffset] = options.offset;

    const shadowColor = options.shadowColor || DEFAULT_FONT_SHADOW_COLOR;

    const numLabels = labels.numRows;

    for (let i = 0; i < numLabels; ++i) {
      const zoomIn = labels.getChild('zoomIn').at(i);
      const zoomOut = labels.getChild('zoomOut').at(i);

      if (zoomIn > zoom || zoom > zoomOut) {
        continue;
      }

      const dataX = labels.getChild('x').at(i);
      const dataY = labels.getChild('y').at(i);

      if (
        dataX < minDataX ||
        dataX > maxDataX ||
        dataY < minDataY ||
        dataY > maxDataY
      ) {
        continue;
      }

      const text = labels.getChild('label').at(i);
      const fontColor = labels.getChild('fontColor').at(i);
      const fontFace = labels.getChild('fontFace').at(i);
      const fontStyle = labels.getChild('fontStyle').at(i);
      const fontWeight = labels.getChild('fontWeight').at(i);
      const fontSize = labels.getChild('fontSize').at(i) * fontScale;
      const zoomFadeExtent = labels.getChild('zoomFadeExtent').at(i);

      // We could also use the zoomFadeExtent to zoom labels in but it appears
      // to be nicer to show them instantly.
      // const alphaFadeIn = Math.max(
      //   0,
      //   Math.min(
      //     1,
      //     (zoom - labelData.zoomLevelIn[i]) /
      //     labelData.zoomFadeExtent[i]
      //   )
      // );

      const alphaFadeOut = Math.max(
        0,
        Math.min(1, (zoomOut - zoom) / zoomFadeExtent),
      );

      ctx.font = `${fontStyle} ${fontWeight} ${fontSize * dpr}px ${fontFace}`;

      drawText(
        text,
        (xScale(dataX) + xOffset) * dpr,
        (yScale(dataY) + yOffset) * dpr,
        fontColor,
        alphaFadeOut,
        shadowColor,
      );
    }
    console.log('Rendering', numLabels, 'took', performance.now() - t0, 'msec');
  };

  return { clear, render };
};
