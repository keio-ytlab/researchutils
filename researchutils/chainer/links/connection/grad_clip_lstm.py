import functools
import operator

from researchutils.chainer.functions.activation import grad_clip_lstm

from chainer.backends import cuda
from chainer.links.connection import lstm
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer import variable


class GradClipLSTM(lstm.LSTM):
    """Fully-connected LSTM layer with gradient clip before each gates.
    See: https://arxiv.org/abs/1308.0850

    For detail description of LSTM layer itself, check original LSTM implementation of chainer.
    """

    def __init__(self, in_size, out_size=None, lateral_init=None,
                 upward_init=None, bias_init=None, forget_bias_init=None,
                 clip_min=None, clip_max=None):
        self.clip_min = clip_min
        self.clip_max = clip_max
        super(GradClipLSTM, self).__init__(
            in_size, out_size, lateral_init, upward_init, bias_init,
            forget_bias_init)
   
    def __call__(self, x):
        """Updates the internal state and returns the LSTM outputs.
        Args:
            x (~chainer.Variable): A new batch from the input sequence.
        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.
        """
        if self.upward.W.data is None:
            with cuda.get_device_from_id(self._device_id):
                in_size = functools.reduce(operator.mul, x.shape[1:], 1)
                self.upward._initialize_params(in_size)
                self._initialize_params()

        batch = x.shape[0]
        lstm_in = self.upward(x)
        h_rest = None
        if self.h is not None:
            h_size = self.h.shape[0]
            if batch == 0:
                h_rest = self.h
            elif h_size < batch:
                msg = ('The batch size of x must be equal to or less than'
                       'the size of the previous state h.')
                raise TypeError(msg)
            elif h_size > batch:
                h_update, h_rest = split_axis.split_axis(
                    self.h, [batch], axis=0)
                lstm_in += self.lateral(h_update)
            else:
                lstm_in += self.lateral(self.h)
        if self.c is None:
            xp = self.xp
            with cuda.get_device_from_id(self._device_id):
                self.c = variable.Variable(
                    xp.zeros((batch, self.state_size), dtype=x.dtype))
        self.c, y = grad_clip_lstm.grad_clip_lstm(
            self.c, lstm_in, clip_min=self.clip_min, clip_max=self.clip_max)

        if h_rest is None:
            self.h = y
        elif len(y.data) == 0:
            self.h = h_rest
        else:
            self.h = concat.concat([y, h_rest], axis=0)

        return y
