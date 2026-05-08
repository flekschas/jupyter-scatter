import type { WritableAtom } from 'jotai';
import type { MouseEvent, SVGProps } from 'react';

/** Anywidget model interface (subset used by toolbar components) */
export interface WidgetModel {
  get(key: string): unknown;
  set(key: string, value: unknown): void;
  on(event: string, handler: () => void): void;
  off(event: string, handler: () => void): void;
  save_changes(): void;
  send(msg: Record<string, unknown>): void;
}

/** The vanilla JS scatter view instance */
export interface ScatterView {
  el: HTMLElement;
  canvas: HTMLCanvasElement | null;
  model: WidgetModel;
  scatterplot: unknown;
  eventTypes: Record<string, string>;
  fullscreenFullWidthHeight: boolean;
  fullscreenWidth: number;
  fullscreenHeight: number;
  fullscreenExportScale: number;
  viewDownload(options?: {
    transparentBackgroundColor?: boolean;
    scale?: number;
  }): void;
  resetView(animation?: number, dataExtent?: boolean): void;
  toggleFullscreen(): void;
  updateContainerDimensions(): void;
  fullscreenContainerStyle(): void;
  getXPadding(): number;
  getYPadding(): number;
}

/** Background color: either float array [r,g,b] or [r,g,b,a] (0-1) or CSS string */
export type BackgroundColor = number[] | string | null;

/** A Jotai atom synced to a widget model property */
export type ModelAtom<T> = WritableAtom<T, [T], void>;

/** All model-synced atoms created by createWidgetStore */
export interface WidgetAtoms {
  mouseMode: ModelAtom<string>;
  cameraIsFixed: ModelAtom<boolean>;
  lassoType: ModelAtom<string>;
  lassoBrushSize: ModelAtom<number>;
  toolbarButtons: ModelAtom<string[]>;
  backgroundColor: ModelAtom<BackgroundColor>;
}

/** SVG icon component props */
export type IconProps = SVGProps<SVGSVGElement>;

/** Button click handler */
export type ButtonClickHandler = (e: MouseEvent<HTMLButtonElement>) => void;
