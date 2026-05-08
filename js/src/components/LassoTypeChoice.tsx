import { useAtomValue, useSetAtom } from 'jotai';
import type { ComponentType } from 'react';
import { useRef, useState } from 'react';

import { useAtoms } from '../hooks/use-widget';
import type { IconProps } from '../types';
import { Button } from './Button';
import { BrushLassoIcon, FreeformLassoIcon, RectangleLassoIcon } from './icons';

interface LassoOption {
  value: string;
  label: string;
  // biome-ignore lint/style/useNamingConvention: Icon is a React component
  Icon: ComponentType<IconProps>;
}

const OPTIONS: LassoOption[] = [
  { value: 'freeform', label: 'Freeform', Icon: FreeformLassoIcon },
  { value: 'brush', label: 'Brush', Icon: BrushLassoIcon },
  { value: 'rectangle', label: 'Rectangle', Icon: RectangleLassoIcon },
];

export function LassoTypeChoice() {
  const atoms = useAtoms();
  const lassoType = useAtomValue(atoms.lassoType);
  const setLassoType = useSetAtom(atoms.lassoType);
  const [open, setOpen] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const CurrentIcon =
    OPTIONS.find((o) => o.value === lassoType)?.Icon ?? FreeformLassoIcon;

  return (
    <div className="jss:relative">
      <Button
        ref={buttonRef}
        icon={<CurrentIcon style={{ width: 18, height: 18 }} />}
        tooltip="Lasso Type"
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
            className={[
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
                className="jss:justify-start! jss:gap-2 jss:px-2 jss:py-1 jss:text-sm jss:whitespace-nowrap jss:border-none"
                onClick={() => {
                  setLassoType(value);
                  setOpen(false);
                }}
              >
                {label}
              </Button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
