import numpy
import chainer.functions as F

from chainer.functions.activation import lstm
from chainer.functions.activation.lstm import _extract_gates


class GradClipLSTM(lstm.LSTM):
    """
    Long short-term memory unit with forget gate and gradient clipping before each gates.
    It has two inputs (c, x) and two outputs (c, h), where c indicates the cell
    state. x must have four times channels compared to the number of units.

    Gradient clipping is done during backward process
    and not before applying the gradient to weights.
    
    See: https://arxiv.org/abs/1308.0850
    """

    def __init__(self, clip_min, clip_max):
        super(GradClipLSTM, self).__init__()
        clip_min = clip_min if clip_min is not None else numpy.finfo(
            numpy.float32).min
        clip_max = clip_max if clip_max is not None else numpy.finfo(
            numpy.float32).max
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def backward(self, inputs, grads):
        gc_prev, gx = super(GradClipLSTM, self).backward(inputs, grads)
        return gc_prev, F.clip(gx, self.clip_min, self.clip_max)


def grad_clip_lstm(c_prev, x, clip_min=None, clip_max=None):
    return GradClipLSTM(clip_min, clip_max).apply((c_prev, x))
