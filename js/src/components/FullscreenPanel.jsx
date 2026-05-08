import { CaretDown, CaretUp } from '@phosphor-icons/react';
import { useCallback, useEffect, useState } from 'react';

import { useScatterView } from '../hooks/use-widget.js';

export function FullscreenPanel() {
  const scatterView = useScatterView();

  const [expanded, setExpanded] = useState(false);
  const [fullWidthHeight, setFullWidthHeight] = useState(true);
  const [exportScale, setExportScale] = useState(1);

  const bBox = scatterView.el.getBoundingClientRect();
  const hasAxes = scatterView.model.get('axes');
  const xPadding = hasAxes ? scatterView.getXPadding() : 0;
  const yPadding = hasAxes ? scatterView.getYPadding() : 0;

  const [widthMax] = useState(() => bBox.width - xPadding);
  const [heightMax] = useState(
    () =>
      bBox.height -
      yPadding -
      Number.parseFloat(getComputedStyle(document.documentElement).fontSize) *
        3,
  );
  const [width, setWidth] = useState(widthMax);
  const [height, setHeight] = useState(heightMax);

  const canvasWidth = scatterView.canvas?.width ?? 0;
  const canvasHeight = scatterView.canvas?.height ?? 0;
  const downloadWidth = canvasWidth * exportScale;
  const downloadHeight = canvasHeight * exportScale;

  const syncToView = useCallback(() => {
    scatterView.fullscreenFullWidthHeight = fullWidthHeight;
    scatterView.fullscreenWidth = width;
    scatterView.fullscreenHeight = height;
    scatterView.fullscreenExportScale = exportScale;
    scatterView.updateContainerDimensions();
    scatterView.fullscreenContainerStyle();
  }, [scatterView, fullWidthHeight, width, height, exportScale]);

  useEffect(() => {
    syncToView();
  }, [syncToView]);

  function handleToggle() {
    if (expanded) {
      setExpanded(false);
      if (!fullWidthHeight) {
        setFullWidthHeight(true);
      }
    } else {
      setExpanded(true);
    }
  }

  function handleFullWidthHeightChange(e) {
    setFullWidthHeight(e.target.checked);
  }

  function handleWidthChange(e) {
    const v = Math.max(1, Math.min(widthMax, Number(e.target.value)));
    setWidth(v);
  }

  function handleHeightChange(e) {
    const v = Math.max(1, Math.min(heightMax, Number(e.target.value)));
    setHeight(v);
  }

  function handleScaleChange(e) {
    setExportScale(Number(e.target.value));
  }

  function handleDownload() {
    scatterView.viewDownload({ scale: exportScale });
  }

  const inputStyle = {
    userSelect: 'auto',
    border: '1px solid var(--jss-input-border)',
    backgroundColor: 'var(--jss-bg-input)',
  };

  const disabledColor = fullWidthHeight
    ? 'var(--jss-text-disabled)'
    : 'var(--jss-text)';

  return (
    <div
      class="jss:relative"
      style={{
        height: expanded ? '3rem' : '0',
        transition: 'height 250ms ease',
      }}
    >
      {/* Toggle button */}
      <button
        type="button"
        onClick={handleToggle}
        class={[
          'jss:absolute jss:left-0 jss:bottom-full',
          'jss:w-9 jss:h-[18px] jss:mx-0.5',
          'jss:flex jss:items-center jss:justify-center',
          'jss:border jss:border-solid jss:border-border',
          'jss:rounded-t-sm! jss:rounded-b-none!',
          'jss:bg-bg jss:text-fg',
          'jss:cursor-pointer',
        ].join(' ')}
      >
        {expanded ? <CaretDown size={12} /> : <CaretUp size={12} />}
      </button>

      {/* Panel content */}
      <div
        class={[
          'jss:relative jss:flex jss:items-center',
          'jss:gap-x-4 jss:px-0.5 jss:py-2',
          'jss:border-t-2 jss:border-solid jss:border-border',
          'jss:select-none',
        ].join(' ')}
      >
        {/* Full Width & Height */}
        <label class="jss:flex jss:items-center jss:gap-x-1">
          Full Width &amp; Height
          <input
            type="checkbox"
            checked={fullWidthHeight}
            onChange={handleFullWidthHeightChange}
          />
        </label>

        {/* Width */}
        <label
          class="jss:flex jss:items-center jss:gap-x-1"
          style={{ color: disabledColor }}
        >
          Width
          <input
            type="number"
            value={width}
            disabled={fullWidthHeight}
            step="1"
            min="1"
            max={widthMax}
            onChange={handleWidthChange}
            style={{
              ...inputStyle,
              color: fullWidthHeight
                ? 'var(--jss-text-disabled)'
                : 'var(--jss-text)',
            }}
          />
        </label>

        {/* Height */}
        <label
          class="jss:flex jss:items-center jss:gap-x-1"
          style={{ color: disabledColor }}
        >
          Height
          <input
            type="number"
            value={height}
            disabled={fullWidthHeight}
            step="1"
            min="1"
            max={heightMax}
            onChange={handleHeightChange}
            style={{
              ...inputStyle,
              color: fullWidthHeight
                ? 'var(--jss-text-disabled)'
                : 'var(--jss-text)',
            }}
          />
        </label>

        {/* Divider */}
        <div
          class="jss:w-0.5 jss:h-7"
          style={{ background: 'var(--jss-border)' }}
        />

        {/* Scale */}
        <label class="jss:flex jss:items-center jss:gap-x-1">
          Scale
          <select
            value={exportScale}
            onChange={handleScaleChange}
            style={{
              backgroundColor: 'var(--jss-bg-input)',
              color: 'var(--jss-text)',
            }}
          >
            {[1, 2, 3].map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        {/* DPR */}
        <label
          class="jss:flex jss:items-center jss:gap-x-1"
          title="Device Pixel Ratio"
          style={{ color: 'var(--jss-text-secondary)' }}
        >
          DPR
          <input
            type="number"
            value={window.devicePixelRatio}
            disabled={true}
            style={{
              ...inputStyle,
              color: 'var(--jss-text-disabled)',
            }}
          />
        </label>

        {/* Download */}
        <button
          type="button"
          onClick={handleDownload}
          class={[
            'jss:border jss:border-solid jss:border-border',
            'jss:rounded-sm! jss:px-3 jss:py-1',
            'jss:bg-bg jss:text-fg',
            'jss:cursor-pointer jss:whitespace-nowrap',
            'jss:hover:bg-bg-hover',
          ].join(' ')}
          title={`Download as PNG at ${downloadWidth}×${downloadHeight} pixels`}
        >
          Download as PNG at {downloadWidth}×{downloadHeight} px
        </button>
      </div>
    </div>
  );
}
