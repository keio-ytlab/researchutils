import numpy as np


def one_hot(indices,
            shape,
            on_value=1,
            off_value=0,
            dtype=None):
    if dtype:
        array_dtype = dtype
    else:
        array_dtype = (np.int8 if isinstance(on_value, int)
                       and isinstance(off_value, int) else np.float32)
    array = np.full(shape=shape, fill_value=off_value, dtype=array_dtype)
    for index in indices:
        array[index, ...] = on_value
    return array
