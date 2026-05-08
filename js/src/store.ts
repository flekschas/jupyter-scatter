import { atom, createStore } from 'jotai';

import type { ModelAtom, ScatterView, WidgetAtoms, WidgetModel } from './types';

/**
 * Creates a Jotai atom that stays synced with an anywidget model property.
 * Read returns the current model value; write calls model.set + save_changes.
 */
export function atomWithModel<T>(
  model: WidgetModel,
  propertyName: string,
): ModelAtom<T> {
  const baseAtom = atom<T>(model.get(propertyName) as T);
  let writing = false;

  const derivedAtom = atom(
    (get) => get(baseAtom),
    (_get, set, newValue: T) => {
      writing = true;
      model.set(propertyName, newValue);
      model.save_changes();
      set(baseAtom, newValue);
      writing = false;
    },
  );

  // Subscribe to model changes (from Python side only)
  derivedAtom.onMount = (setAtom) => {
    const handler = () => {
      if (writing) {
        return;
      }
      try {
        setAtom(model.get(propertyName) as T);
      } catch {
        // Model may be disposed during widget teardown
      }
    };
    model.on(`change:${propertyName}`, handler);
    return () => {
      try {
        model.off(`change:${propertyName}`, handler);
      } catch {
        // Model may already be disposed
      }
    };
  };

  return derivedAtom as ModelAtom<T>;
}

/**
 * Atom to hold the scatterView instance (non-reactive, set once).
 */
export const scatterViewAtom = atom<ScatterView | null>(null);

/**
 * Creates a Jotai store pre-populated with model-synced atoms for the toolbar.
 */
export function createWidgetStore(
  model: WidgetModel,
  scatterView: ScatterView,
) {
  const store = createStore();

  store.set(scatterViewAtom, scatterView);

  const atoms: WidgetAtoms = {
    mouseMode: atomWithModel<string>(model, 'mouse_mode'),
    cameraIsFixed: atomWithModel<boolean>(model, 'camera_is_fixed'),
    lassoType: atomWithModel<string>(model, 'lasso_type'),
    lassoBrushSize: atomWithModel<number>(model, 'lasso_brush_size'),
    toolbarButtons: atomWithModel<string[]>(model, 'toolbar_buttons'),
    backgroundColor: atomWithModel(model, 'background_color'),
  };

  return { store, atoms };
}
