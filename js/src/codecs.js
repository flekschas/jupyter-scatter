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

class NumpyCodec {
  /** @param {keyof typeof DTYPES} dtype */
  constructor(dtype) {
    if (!(dtype in DTYPES)) {
      throw Error(`Dtype not supported, got ${JSON.stringify(dtype)}.`);
    }
    this.dtype = dtype;
  }
}

class Numpy2D extends NumpyCodec {
  /**
   * @param {SerializedArray<[number, number]>} data
   * @returns {number[][] | null}
   */
  deserialize(data) {
    if (data == null) return null;
    // Take full view of data buffer
    const arr = new DTYPES[this.dtype](data.view.buffer);
    // Chunk single TypedArray into nested Array of points
    const [height, width] = data.shape;
    // Float32Array(width * height) -> [Array(width), Array(width), ...]
    const points = Array.from({ length: height }).map((_, i) =>
      Array.from(arr.subarray(i * width, (i + 1) * width))
    );
    return points;
  }

  /**
   * @param {number[][]} data
   * @returns {SerializedArray<[number, number]>}
   */
  serialize(data) {
    const height = data.length;
    const width = data[0].length;
    const arr = new DTYPES[this.dtype](height * width);
    for (let i = 0; i < data.length; i++) {
      arr.set(data[i], i * height);
    }
    return {
      view: new DataView(arr.buffer),
      dtype: this.dtype,
      shape: [height, width],
    };
  }
}

class Numpy1D extends NumpyCodec {
  /**
   * @param {SerializedArray<[number]>} data
   * @returns {number[] | null}
   */
  deserialize(data) {
    if (data == null) return null;
    // for some reason can't be a typed array
    return Array.from(new DTYPES[this.dtype](data.view.buffer));
  }

  /**
   * @param {number[]} data
   * @returns {SerializedArray<[number]>}
   */
  serialize(data) {
    const arr = new DTYPES[this.dtype](data);
    return {
      view: new DataView(arr.buffer),
      dtype: this.dtype,
      shape: [data.length],
    };
  }
}

module.exports = { Numpy1D, Numpy2D };
