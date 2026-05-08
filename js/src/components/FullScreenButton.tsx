import { FrameCornersIcon, ResizeIcon } from '@phosphor-icons/react';
import { useEffect, useState } from 'react';

import { useScatterView } from '../hooks/use-widget';
import { Button } from './Button';

export function FullScreenButton() {
  const scatterView = useScatterView();
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    function handler() {
      setIsFullscreen(!!document.fullscreenElement);
    }

    document.addEventListener('fullscreenchange', handler);
    return () => document.removeEventListener('fullscreenchange', handler);
  }, []);

  return (
    <Button
      icon={
        isFullscreen ? <ResizeIcon size={18} /> : <FrameCornersIcon size={18} />
      }
      tooltip="Full Screen"
      onClick={() => scatterView.toggleFullscreen()}
    />
  );
}
