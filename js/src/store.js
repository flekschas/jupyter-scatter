import { atom, createStore } from 'jotai';

/**
 * Creates a Jotai atom that stays synced with an anywidget model property.
 * Read returns the current model value; write calls model.set + save_changes.
 */
export function atomWithModel(model, propertyName) {
  const baseAtom = atom(model.get(propertyName));
  let writing = false;

  const derivedAtom = atom(
    (get) => get(baseAtom),
    (_get, set, newValue) => {
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
      if (writing) { return; }
      try {
        setAtom(model.get(propertyName));
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
      backgroundColor: atomWithModel(model, 'background_color'),
    },
  };
}
