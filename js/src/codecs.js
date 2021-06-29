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
   * @param {{buffer: DataView, dtype: keyof typeof DTYPES, shape: [number, number]}} data
   * @returns {number[][]}
   */
  deserialize(data) {
    if (data == null) return null;
    // Take full view of data buffer
    const arr = new DTYPES[this.dtype](data.buffer.buffer);
    // Chunk single TypedArray into nested Array of points
    const [height, width] = data.shape;
    // Float32Array(width * height) -> [Array(width), Array(width), ...]
    const points = Array
      .from({ length: height })
      .map((_, i) => Array.from(arr.subarray(i * width, (i + 1) * width)));
    return points;
  }

  /**
   * @param {number[][]} data
   * @returns {{data: ArrayBuffer, dtype: keyof typeof DTYPES, shape: [number, number]}}
   */
  serialize(data) {
    const height = data.length;
    const width = data[0].length;
    const arr = new DTYPES[this.dtype](height * width);
    for (let i = 0; i < data.length; i++) {
      arr.set(data[i], i * height);
    }
    return { data: arr.buffer, dtype: this.dtype, shape: [height, width] };
  }
}

class Numpy1D extends NumpyCodec {

  /**
   * @param {{buffer: DataView, dtype: keyof typeof DTYPES, shape: [number]}} data
   * @returns {number[]}
   */
  deserialize(data) {
    if (data == null) return null;
    // for some reason can't be a typed array
    return Array.from(new DTYPES[this.dtype](data.buffer.buffer));
  }

  /**
   * @param {number[]} data
   * @returns {{data: ArrayBuffer, dtype: keyof typeof DTYPES, shape: [number]}}
   */
  serialize(data) {
    const arr = new DTYPES[this.dtype](data)
    return { data: arr.buffer, dtype: this.dtype, shape: [data.length] };
  }
}

module.exports = { Numpy1D, Numpy2D };
