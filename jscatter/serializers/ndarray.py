import numpy as np


# Code extracted from maartenbreddels ipyvolume
def array_to_binary(ar, obj=None, force_contiguous=True):
    if ar is None:
        return None
    if ar.dtype.kind not in ['u', 'i', 'f']:  # ints and floats
        raise ValueError('unsupported dtype: %s' % (ar.dtype))
    if ar.dtype == np.float64:  # WebGL does not support float64, case it here
        ar = ar.astype(np.float32)
    if ar.dtype == np.int64:  # JS does not support int64
        ar = ar.astype(np.int32)
    if force_contiguous and not ar.flags['C_CONTIGUOUS']:  # make sure it's contiguous
        ar = np.ascontiguousarray(ar)
    return {'view': memoryview(ar), 'dtype': str(ar.dtype), 'shape': ar.shape}


def binary_to_array(value, obj=None):
    if value is None:
        return None
    return np.frombuffer(value['view'], dtype=value['dtype']).reshape(value['shape'])


serialization = dict(to_json=array_to_binary, from_json=binary_to_array)
