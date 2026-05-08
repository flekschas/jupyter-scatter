import { DownloadSimpleIcon } from '@phosphor-icons/react';

import { useScatterView } from '../hooks/use-widget';
import { Button } from './Button';

export function DownloadButton() {
  const scatterView = useScatterView();

  return (
    <Button
      icon={<DownloadSimpleIcon size={18} />}
      tooltip="Download View as PNG"
      onClick={(e) => {
        scatterView.viewDownload({
          transparentBackgroundColor: e.altKey,
        });
      }}
    />
  );
}
