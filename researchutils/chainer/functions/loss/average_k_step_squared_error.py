import chainer.functions as F


def average_k_step_squared_error(x1, x2, k_step):
    """Average k-step squared error introduced by Oh et al.
    See: https://arxiv.org/abs/1507.08750
    Args:
        x1 (array): predicted image
        x2 (array): expected image
        k_step (int): maximum steps to predict from given input
    """
    return F.sum(F.squared_error(x1, x2)) / k_step / 2
