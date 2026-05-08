import { Provider, useAtomValue } from 'jotai';
import { Fragment, useMemo } from 'react';

import { AtomsContext } from '../hooks/use-widget';
import { createWidgetStore } from '../store';
import type {
  BackgroundColor,
  ScatterView,
  WidgetAtoms,
  WidgetModel,
} from '../types';
import { BrushSizeSlider } from './BrushSizeSlider';
import { Divider } from './Divider';
import { DownloadButton } from './DownloadButton';
import { FullScreenButton } from './FullScreenButton';
import { LassoTypeChoice } from './LassoTypeChoice';
import { MouseModeToggle } from './MouseModeToggle';
import { ResetButton } from './ResetButton';
import { SaveButton } from './SaveButton';

import '../styles.built.css';

/**
 * Compute relative luminance from a background color value.
 * Returns 'dark' if the background is dark, 'light' otherwise.
 */
function getTheme(bgColor: BackgroundColor): 'light' | 'dark' {
  let r: number;
  let g: number;
  let b: number;

  if (Array.isArray(bgColor)) {
    [r, g, b] = bgColor.map((c) => c * 255);
  } else if (typeof bgColor === 'string') {
    const hex = bgColor.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
    if (hex) {
      r = Number.parseInt(hex[1], 16);
      g = Number.parseInt(hex[2], 16);
      b = Number.parseInt(hex[3], 16);
    } else {
      return 'light';
    }
  } else {
    return 'light';
  }

  const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return luminance < 0.5 ? 'dark' : 'light';
}

function toBgCss(bgColor: BackgroundColor): string | undefined {
  if (Array.isArray(bgColor)) {
    return bgColor.length === 4
      ? `rgba(${bgColor
          .slice(0, 3)
          .map((c) => c * 255)
          .join(',')}, ${bgColor[3]})`
      : `rgb(${bgColor.map((c) => c * 255).join(',')})`;
  }
  return bgColor || undefined;
}

const BUTTON_MAP: Record<string, () => React.ReactNode> = {
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

function ToolbarInner({ atoms }: { atoms: WidgetAtoms }) {
  const buttons = useAtomValue(atoms.toolbarButtons);
  const bgColor = useAtomValue(atoms.backgroundColor);
  const theme = useMemo(() => getTheme(bgColor), [bgColor]);
  const bgCss = useMemo(() => toBgCss(bgColor), [bgColor]);

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

interface ToolbarProps {
  model: WidgetModel;
  scatterView: ScatterView;
}

export function Toolbar({ model, scatterView }: ToolbarProps) {
  const { store, atoms } = createWidgetStore(model, scatterView);

  return (
    <Provider store={store}>
      <AtomsContext.Provider value={atoms}>
        <ToolbarInner atoms={atoms} />
      </AtomsContext.Provider>
    </Provider>
  );
}
