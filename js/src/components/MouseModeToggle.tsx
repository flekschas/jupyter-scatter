import { ArrowsOutCardinalIcon, LassoIcon } from '@phosphor-icons/react';
import { useAtomValue, useSetAtom } from 'jotai';
import type { ComponentType } from 'react';

import { useAtoms } from '../hooks/use-widget';
import { Button } from './Button';

const ICONS: Record<string, ComponentType<{ size: number }>> = {
  panZoom: ArrowsOutCardinalIcon,
  lasso: LassoIcon,
};

const TOOLTIPS: Record<string, string> = {
  panZoom: 'Activate pan & zoom',
  lasso: 'Activate lasso selection',
};

interface MouseModeToggleProps {
  mode: 'panZoom' | 'lasso';
}

export function MouseModeToggle({ mode }: MouseModeToggleProps) {
  const atoms = useAtoms();
  const mouseMode = useAtomValue(atoms.mouseMode);
  const setMouseMode = useSetAtom(atoms.mouseMode);
  const cameraIsFixed = useAtomValue(atoms.cameraIsFixed);

  const Icon = ICONS[mode];

  return (
    <Button
      icon={<Icon size={18} />}
      tooltip={TOOLTIPS[mode]}
      active={mouseMode === mode}
      disabled={mode === 'panZoom' && cameraIsFixed}
      onClick={() => setMouseMode(mode)}
    />
  );
}
