import { ArrowsOutCardinalIcon, LassoIcon } from '@phosphor-icons/react';
import { useAtomValue, useSetAtom } from 'jotai';

import { useAtoms } from '../hooks/use-widget.js';
import { Button } from './Button.jsx';

const ICONS = {
  panZoom: ArrowsOutCardinalIcon,
  lasso: LassoIcon,
};

const TOOLTIPS = {
  panZoom: 'Activate pan & zoom',
  lasso: 'Activate lasso selection',
};

export function MouseModeToggle({ mode }) {
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
