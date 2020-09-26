import nnabla.functions as F


def average_k_step_squared_error(x1, x2, k_step):
    """
    Average k-step squared error introduced by Oh et al.

    .. math::

       \\frac{1}{2K}\sum_{i}\sum_{t}\sum_{k}\|\hat{\mathbf{x}}_{t+k}^{(i)} - \mathbf{x}_{t+k}^{(i)}\|^{2}

    See: https://arxiv.org/abs/1507.08750

    Parameters
    -------
    x1 : array
        predicted image
    x2 : array
        expected image
    k_step : int
        maximum steps to predict from given input

    Returns
    -------
    error : Variable
        k-step squared error
    """
    return (0.5 * F.sum(F.squared_error(x1, x2))) / k_step


def copy(x):
    return F.reshape(x, shape=x.shape, inplace=False)
