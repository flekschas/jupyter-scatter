import { atom, createStore } from 'jotai';

/**
 * Creates a Jotai atom that stays synced with an anywidget model property.
 * Read returns the current model value; write calls model.set + save_changes.
 */
export function atomWithModel(model, propertyName) {
  const baseAtom = atom(model.get(propertyName));

  const derivedAtom = atom(
    (get) => get(baseAtom),
    (_get, set, newValue) => {
      model.set(propertyName, newValue);
      model.save_changes();
      set(baseAtom, newValue);
    },
  );

  // Subscribe to model changes (from Python side)
  derivedAtom.onMount = (setAtom) => {
    const handler = () => setAtom(model.get(propertyName));
    model.on(`change:${propertyName}`, handler);
    return () => model.off(`change:${propertyName}`, handler);
  };

  return derivedAtom;
}

/**
 * Atom to hold the scatterView instance (non-reactive, set once).
 */
export const scatterViewAtom = atom(null);

/**
 * Creates a Jotai store pre-populated with model-synced atoms for the toolbar.
 */
export function createWidgetStore(model, scatterView) {
  const store = createStore();

  store.set(scatterViewAtom, scatterView);

  return {
    store,
    atoms: {
      mouseMode: atomWithModel(model, 'mouse_mode'),
      cameraIsFixed: atomWithModel(model, 'camera_is_fixed'),
      lassoType: atomWithModel(model, 'lasso_type'),
      lassoBrushSize: atomWithModel(model, 'lasso_brush_size'),
      toolbarButtons: atomWithModel(model, 'toolbar_buttons'),
    },
  };
}
