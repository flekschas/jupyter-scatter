import { ArrowsClockwiseIcon } from '@phosphor-icons/react';

import { useScatterView } from '../hooks/use-widget';
import { Button } from './Button';

export function ResetButton() {
  const scatterView = useScatterView();

  return (
    <Button
      icon={<ArrowsClockwiseIcon size={18} />}
      tooltip="Reset View"
      onClick={(e) => {
        scatterView.resetView(500, e.altKey);
      }}
    />
  );
}
