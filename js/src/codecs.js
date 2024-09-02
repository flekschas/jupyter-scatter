import { camelToSnake, snakeToCamel } from './utils';

const DTYPES = {
  uint8: Uint8Array,
  int8: Int8Array,
  uint16: Uint16Array,
  int16: Int16Array,
  uint32: Uint32Array,
  int32: Int32Array,
  float32: Float32Array,
  float64: Float64Array,
};

/**
 * @template {number[]} Shape
 * @typedef SerializedArray
 * @prop {DataView} view
 * @prop {keyof typeof DTYPES} dtype
 * @prop {Shape} shape
 */
export function NumpyImage() {
  return {
    /**
     * @param {SerializedArray<[number, number, number]>} data
     * @returns {ImageData | null}
     */
    deserialize(data) {
      if (data == null) return null;
      // Take full view of data buffer
      const dataArray = new Uint8ClampedArray(data.view.buffer);
      // Chunk single TypedArray into nested Array of points
      const [height, width] = data.shape;
      return new ImageData(dataArray, width, height);
    },
    /**
     * @param {ImageData} data
     * @returns {SerializedArray<[number, number, number]>}
     */
    serialize(imageData) {
      return {
        view: new DataView(imageData.data.buffer),
        dtype: 'uint8',
        shape: [imageData.height, imageData.width, 4],
      };
    }
  }
}

/**
 * @template {number[]} Shape
 * @typedef SerializedArray
 * @prop {DataView} view
 * @prop {keyof typeof DTYPES} dtype
 * @prop {Shape} shape
 */
export function Numpy2D(dtype) {
  if (!(dtype in DTYPES)) {
    throw Error(`Dtype not supported, got ${JSON.stringify(dtype)}.`);
  }
  return {
    /**
     * @param {SerializedArray<[number, number]>} data
     * @returns {number[][] | null}
     */
    deserialize(data) {
      if (data == null) return null;
      // Take full view of data buffer
      const arr = new DTYPES[dtype](data.view.buffer);
      // Chunk single TypedArray into nested Array of points
      const [height, width] = data.shape;
      // Float32Array(width * height) -> [Array(width), Array(width), ...]
      const points = Array.from({ length: height }).map((_, i) =>
        Array.from(arr.subarray(i * width, (i + 1) * width))
      );
      return points;
    },
    /**
     * @param {number[][]} data
     * @returns {SerializedArray<[number, number]>}
     */
    serialize(data) {
      const height = data.length;
      const width = data[0].length;
      const arr = new DTYPES[dtype](height * width);
      for (let i = 0; i < data.length; i++) {
        arr.set(data[i], i * height);
      }
      return {
        view: new DataView(arr.buffer),
        dtype: dtype,
        shape: [height, width],
      };
    }
  }
}

export function Numpy1D(dtype) {
  if (!(dtype in DTYPES)) {
    throw Error(`Dtype not supported, got ${JSON.stringify(dtype)}.`);
  }
  return {
    /**
     * @param {SerializedArray<[number]>} data
     * @returns {number[] | null}
     */
    deserialize(data) {
      if (data == null) return null;
      // for some reason can't be a typed array
      return Array.from(new DTYPES[dtype](data.view.buffer));
    },
    /**
     * @param {number[]} data
     * @returns {SerializedArray<[number]>}
     */
    serialize(data) {
      const arr = new DTYPES[dtype](data);
      return {
        view: new DataView(arr.buffer),
        dtype: dtype,
        shape: [data.length],
      };
    }
  }
}

export function Annotations() {
  const pyToJsKey = { x_start: 'x1', x_end: 'x2', y_start: 'y1', y_end: 'y2' };
  const jsToPyKey = { x1: 'x_start', x2: 'x_end', y1: 'y_start', y2: 'y_end' };
  return {
    /**
     * @param {string[] | null} annotationStrs
     * @returns {object[]}
     */
    deserialize: (annotationStrs) => {
      if (annotationStrs === null) {
        return null;
      }
      return annotationStrs.map((annotationStr) => {
        return Object.entries(JSON.parse(annotationStr)).reduce((acc, [key, value]) => {
          const jsKey = key in pyToJsKey ? pyToJsKey[key] : snakeToCamel(key);
          acc[jsKey] = value;
          return acc;
        }, {})
      });
    },
    /**
     * @param {object[] | null} annotations
     * @returns {string[]}
     */
    serialize: (annotations) => {
      if (annotations === null) {
        return null;
      }
      return annotations.map((annotation) => {
        const pyAnnotation = Object.entries(annotation).reduce((acc, [key, value]) => {
          const pyKey = key in jsToPyKey ? jsToPyKey[key] : camelToSnake(key);
          acc[pyKey] = value;
          return acc;
        }, {});
        return JSON.stringify(pyAnnotation);
      })
    },
  }
}
