import { useAtomValue, useSetAtom } from 'jotai';
import { useEffect, useRef, useState } from 'react';

import { useAtoms } from '../hooks/use-widget.js';
import { Button } from './Button.jsx';
import {
  BrushLassoIcon,
  FreeformLassoIcon,
  RectangleLassoIcon,
} from './icons.jsx';

const OPTIONS = [
  // biome-ignore lint/style/useNamingConvention: Icon is a React component
  { value: 'freeform', label: 'Freeform', Icon: FreeformLassoIcon },
  // biome-ignore lint/style/useNamingConvention: Icon is a React component
  { value: 'brush', label: 'Brush', Icon: BrushLassoIcon },
  // biome-ignore lint/style/useNamingConvention: Icon is a React component
  { value: 'rectangle', label: 'Rectangle', Icon: RectangleLassoIcon },
];

export function LassoTypeChoice() {
  const atoms = useAtoms();
  const lassoType = useAtomValue(atoms.lassoType);
  const setLassoType = useSetAtom(atoms.lassoType);
  const [open, setOpen] = useState(false);
  const dialogRef = useRef(null);
  const buttonRef = useRef(null);

  const CurrentIcon =
    OPTIONS.find((o) => o.value === lassoType)?.Icon ?? FreeformLassoIcon;

  useEffect(() => {
    if (!open) {
      return;
    }

    function handleClickOutside(e) {
      if (
        dialogRef.current &&
        !dialogRef.current.contains(e.target) &&
        !buttonRef.current.contains(e.target)
      ) {
        setOpen(false);
      }
    }

    document.addEventListener('pointerdown', handleClickOutside);
    return () =>
      document.removeEventListener('pointerdown', handleClickOutside);
  }, [open]);

  return (
    <div class="jss:relative">
      <Button
        ref={buttonRef}
        icon={<CurrentIcon style={{ width: 18, height: 18 }} />}
        tooltip="Lasso Type"
        onClick={() => setOpen((prev) => !prev)}
      />

      {open && (
        <div
          ref={dialogRef}
          class={[
            'jss:absolute jss:left-full jss:top-0 jss:ml-1 jss:z-50',
            'jss:bg-bg jss:border jss:border-solid jss:border-border',
            'jss:rounded-sm! jss:shadow-md',
            'jss:p-0.5 jss:flex jss:flex-col jss:gap-0.5',
          ].join(' ')}
        >
          {OPTIONS.map(({ value, label, Icon }) => (
            <Button
              key={value}
              icon={<Icon style={{ width: 18, height: 18, flexShrink: 0 }} />}
              tooltip={label}
              active={value === lassoType}
              class="jss:justify-start! jss:gap-2 jss:px-2 jss:py-1 jss:text-sm jss:whitespace-nowrap jss:border-none"
              onClick={() => {
                setLassoType(value);
                setOpen(false);
              }}
            >
              {label}
            </Button>
          ))}
        </div>
      )}
    </div>
  );
}
