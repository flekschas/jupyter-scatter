import { useAtomValue } from 'jotai';
import { createContext, useContext } from 'react';

import { scatterViewAtom } from '../store.js';

/**
 * Context to pass the atoms object down without prop drilling.
 */
export const AtomsContext = createContext(null);

export const useAtoms = () => useContext(AtomsContext);
export const useScatterView = () => useAtomValue(scatterViewAtom);
