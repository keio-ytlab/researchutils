import numpy as np


def one_hot(indices,
            shape,
            on_value=1,
            off_value=0,
            dtype=None):
    """
    Create one hot vector(s)

    Parameters
    -------
    indices : list of int
        list of index to set on_value in each vector
    shape : tuple of int
        Number of rows and arrays to create. Length of the shape must be 2.
        shape must be in the form of (rows, array_num) 
        and will create array_num vectors of shape (rows, 1)
    on_value : int, default 1
        value to set on given indices
    off_value : int, default 0
        value to set on fields other than given indices
    dtype: numpy.dtype
    
    Returns
    -------
    one hot vectors : numpy.ndarray
        matrix of given shape
    """ 

    if not len(shape) == 2:
        raise ValueError('shape must have only rows and columns size(Should be a matrix or vector)')
    if dtype:
        array_dtype = dtype
    else:
        array_dtype = (np.int8 if isinstance(on_value, int)
                       and isinstance(off_value, int) else np.float32)
    array = np.full(shape=shape, fill_value=off_value, dtype=array_dtype)
    for column, row in enumerate(indices):
        array[row, column] = on_value
    return array


def unzip(zipped_array):
    """
    Do the opposite of zip() to convert array of tuples to tuple of arrays 

    Parameters
    -------
    zipped array : array-like of tuple

    Returns
    -------
    unzipped tuple : tuple of array-like
    """ 

    return zip(*zipped_array)


