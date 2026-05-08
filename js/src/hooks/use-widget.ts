import { useAtomValue } from 'jotai';
import { createContext, useContext } from 'react';

import { scatterViewAtom } from '../store';

import type { ScatterView, WidgetAtoms } from '../types';

/**
 * Context to pass the atoms object down without prop drilling.
 */
export const AtomsContext = createContext<WidgetAtoms | null>(null);

export const useAtoms = (): WidgetAtoms => {
  const atoms = useContext(AtomsContext);
  if (!atoms) {
    throw new Error('useAtoms must be used within a Toolbar');
  }
  return atoms;
};

export const useScatterView = (): ScatterView => {
  const view = useAtomValue(scatterViewAtom);
  if (!view) {
    throw new Error('useScatterView must be used within a Toolbar');
  }
  return view;
};
