import { Provider, useAtomValue } from 'jotai';
import { Fragment, useMemo } from 'react';

import { AtomsContext } from '../hooks/use-widget.js';
import { createWidgetStore } from '../store.js';

/**
 * Compute relative luminance from a background color value.
 * Returns 'dark' if the background is dark, 'light' otherwise.
 */
function getTheme(bgColor) {
  let r;
  let g;
  let b;

  if (Array.isArray(bgColor)) {
    // Float array [0-1] from regl-scatterplot
    [r, g, b] = bgColor.map((c) => c * 255);
  } else if (typeof bgColor === 'string') {
    // Try to parse hex or rgb string
    const hex = bgColor.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
    if (hex) {
      r = Number.parseInt(hex[1], 16);
      g = Number.parseInt(hex[2], 16);
      b = Number.parseInt(hex[3], 16);
    } else {
      // Default to light if we can't parse
      return 'light';
    }
  } else {
    return 'light';
  }

  // Relative luminance (ITU-R BT.709)
  const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return luminance < 0.5 ? 'dark' : 'light';
}
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
  const bgColor = useAtomValue(atoms.backgroundColor);
  const theme = useMemo(() => getTheme(bgColor), [bgColor]);

  const bgCss = Array.isArray(bgColor)
    ? bgColor.length === 4
      ? `rgba(${bgColor
          .slice(0, 3)
          .map((c) => c * 255)
          .join(',')}, ${bgColor[3]})`
      : `rgb(${bgColor.map((c) => c * 255).join(',')})`
    : bgColor || undefined;

  return (
    <div
      data-jss-theme={theme}
      className="jss:flex jss:flex-col jss:items-stretch jss:w-9 jss:h-full jss:gap-0.5 jss:p-1"
      style={{ backgroundColor: bgCss }}
    >
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
