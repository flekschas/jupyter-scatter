import { CameraIcon } from '@phosphor-icons/react';

import { useScatterView } from '../hooks/use-widget.js';
import { Button } from './Button.jsx';

export function SaveButton() {
  const scatterView = useScatterView();

  return (
    <Button
      icon={<CameraIcon size={18} />}
      tooltip="Save View to Widget Property"
      onClick={(e) => {
        scatterView.model.send({
          type: scatterView.eventTypes.VIEW_SAVE,
          transparentBackgroundColor: e.altKey,
        });
      }}
    />
  );
}
