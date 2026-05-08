import { useAtomValue, useSetAtom } from 'jotai';
import { useCallback, useRef, useState } from 'react';

import { useAtoms } from '../hooks/use-widget';
import { Button } from './Button';
import { BrushSizeIcon } from './icons';

const BRUSH_SIZE_MIN = 1;
const BRUSH_SIZE_MAX = 128;

export function BrushSizeSlider() {
  const atoms = useAtoms();
  const lassoType = useAtomValue(atoms.lassoType);
  const brushSize = useAtomValue(atoms.lassoBrushSize);
  const setBrushSize = useSetAtom(atoms.lassoBrushSize);
  const [open, setOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [circlePos, setCirclePos] = useState({ top: 0, left: 0 });
  const buttonRef = useRef<HTMLButtonElement>(null);
  const sliderRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  const updateCircle = useCallback(() => {
    if (!sliderRef.current || !dialogRef.current || !containerRef.current) {
      return;
    }

    const containerBBox = containerRef.current.getBoundingClientRect();
    const sliderBBox = sliderRef.current.getBoundingClientRect();

    const thumbRadius = 8;
    const trackWidth = sliderBBox.width - thumbRadius * 2;
    const relValue =
      (Number(sliderRef.current.value) - Number(sliderRef.current.min)) /
      (Number(sliderRef.current.max) - Number(sliderRef.current.min));

    setCirclePos({
      top: sliderBBox.top - containerBBox.top + sliderBBox.height / 2,
      left:
        sliderBBox.left -
        containerBBox.left +
        thumbRadius +
        trackWidth * relValue,
    });
  }, []);

  const handleInput = useCallback(
    (e: React.SyntheticEvent<HTMLInputElement>) => {
      setBrushSize(Number(e.currentTarget.value));
      updateCircle();
    },
    [setBrushSize, updateCircle],
  );

  const handlePointerDown = useCallback(() => {
    setDragging(true);
    if (sliderRef.current) {
      setBrushSize(Number(sliderRef.current.value));
      updateCircle();
    }
  }, [setBrushSize, updateCircle]);

  const handlePointerUp = useCallback(() => {
    setDragging(false);
  }, []);

  if (lassoType !== 'brush') {
    return null;
  }

  return (
    <div className="jss:relative" ref={containerRef}>
      <Button
        ref={buttonRef}
        icon={<BrushSizeIcon style={{ width: 18, height: 18 }} />}
        tooltip="Brush Size"
        onClick={() => setOpen((prev) => !prev)}
      />

      {open && (
        <>
          {/* biome-ignore lint/a11y/useKeyWithClickEvents: backdrop is mouse-only dismiss */}
          <div
            className="jss:fixed jss:inset-0 jss:z-40"
            onClick={() => setOpen(false)}
          />
          <div
            ref={dialogRef}
            className={[
              'jss:absolute jss:left-full jss:top-0 jss:ml-1 jss:z-50',
              'jss:bg-bg jss:border jss:border-solid jss:border-border',
              'jss:rounded-sm! jss:shadow-md',
              'jss:h-7 jss:px-2 jss:flex jss:items-center jss:gap-2',
            ].join(' ')}
          >
            <label className="jss:text-sm jss:text-fg jss:whitespace-nowrap jss:flex jss:items-center jss:gap-2">
              Brush Size
              <input
                ref={sliderRef}
                type="range"
                min={BRUSH_SIZE_MIN}
                max={BRUSH_SIZE_MAX}
                step={1}
                value={brushSize}
                onInput={handleInput}
                onPointerDown={handlePointerDown}
                onPointerUp={handlePointerUp}
                className="jss:w-32"
              />
            </label>
            <span className="jss:text-sm jss:text-fg-secondary jss:w-10 jss:text-right jss:tabular-nums">
              {brushSize}px
            </span>
          </div>
        </>
      )}
      {/* Dashed circle preview — outside the popover so it can't be clipped */}
      <div
        className="jss:absolute jss:-translate-x-1/2 jss:-translate-y-1/2 jss:opacity-0 jss:border jss:border-dashed jss:border-accent jss:rounded-full jss:pointer-events-none jss:user-select-none jss:transition-opacity jss:ease-out jss:duration-150 jss:z-50"
        style={{
          top: circlePos.top,
          left: circlePos.left,
          width: brushSize,
          height: brushSize,
          opacity: dragging ? 0.66 : 0,
        }}
      />
    </div>
  );
}
