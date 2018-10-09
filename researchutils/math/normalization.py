import math
import numpy as np

def normalize_by_min_max(data, min_value=0.0, max_value=1.0):
    """ Normalize the data in min_value and max_value

    Parameters
    -----------
    data : array-like
    min_value : float
        min_value of the fixed data, default is 0.0
    max_value : float
        max_value of the fixed data, default is 1.0
   
    Returns
    ----------
    normalized_data : numpy.ndarray
        the shape of array is same as the input data
    """
    if min_value >= max_value:
        raise ValueError("max angle must be greater than min angle")

    data = np.array(data)
    data_shape = data.shape

    data = data.flatten()

    max_data = np.amax(data)
    min_data = np.amin(data)

    normalized_data = ((data - min_data) / (max_data - min_data)) * (max_value - min_value) + min_value

    return normalized_data.reshape(data_shape)

def standardize(data):
    """Standardize the data. Fixed data's average is 0 and, variance is 1.0

    Parameters
    ------------
    data : array-like
    
    Returns
    -----------
    standarized data : numpy.ndarray
        the shape of array is same as the input data
    """
    data = np.array(data)
    data_shape = data.shape

    data = data.flatten()

    ave_data = np.average(data)
    var_data = np.sqrt(np.var(data))

    standardized_data = (data - ave_data) / var_data
    
    return standardized_data.reshape(data_shape)





