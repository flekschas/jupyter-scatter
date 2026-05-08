import { Provider, useAtomValue } from 'jotai';
import { Fragment } from 'react';

import { AtomsContext } from '../hooks/use-widget.js';
import { createWidgetStore } from '../store.js';
import { BrushSizeSlider } from './BrushSizeSlider.jsx';
import { Divider } from './Divider.jsx';
import { DownloadButton } from './DownloadButton.jsx';
import { FullScreenButton } from './FullScreenButton.jsx';
import { LassoTypeChoice } from './LassoTypeChoice.jsx';
import { MouseModeToggle } from './MouseModeToggle.jsx';
import { ResetButton } from './ResetButton.jsx';
import { SaveButton } from './SaveButton.jsx';

import '../styles.built.css';

const BUTTON_MAP = {
  // biome-ignore lint/style/useNamingConvention: matches Python traitlet value
  pan_zoom: () => <MouseModeToggle mode="panZoom" />,
  lasso: () => <MouseModeToggle mode="lasso" />,
  // biome-ignore lint/style/useNamingConvention: matches Python traitlet value
  lasso_type: () => <LassoTypeChoice />,
  // biome-ignore lint/style/useNamingConvention: matches Python traitlet value
  lasso_brush_size: () => <BrushSizeSlider />,
  divider: () => <Divider />,
  // biome-ignore lint/style/useNamingConvention: matches Python traitlet value
  full_screen: () => <FullScreenButton />,
  save: () => <SaveButton />,
  download: () => <DownloadButton />,
  reset: () => <ResetButton />,
};

function ToolbarInner({ atoms }) {
  const buttons = useAtomValue(atoms.toolbarButtons);

  return (
    <div class="jss:flex jss:flex-col jss:items-stretch jss:w-8 jss:gap-0.5 jss:mx-0.5">
      {buttons.map((name) => {
        const createButton = BUTTON_MAP[name];
        if (!createButton) {
          return null;
        }
        return <Fragment key={name}>{createButton()}</Fragment>;
      })}
    </div>
  );
}

export function Toolbar({ model, scatterView }) {
  const { store, atoms } = createWidgetStore(model, scatterView);

  return (
    <Provider store={store}>
      <AtomsContext.Provider value={atoms}>
        <ToolbarInner atoms={atoms} />
      </AtomsContext.Provider>
    </Provider>
  );
}
