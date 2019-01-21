import functools
import operator

from chainer.backends import cuda
from chainer.functions.activation import lstm as lstm_activation
from chainer import initializers
from chainer.links.connection import linear
from chainer.links.connection import lstm

from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer import variable

class ActionConditionedLSTM(lstm.LSTM):
    """
    LSTM layer which combines input and internal state with given action
    See: https://arxiv.org/pdf/1704.02254.pdf
    """

    def __init__(self, in_size, out_size=None, lateral_init=None,
                 upward_init=None, bias_init=None, forget_bias_init=None,
                 h_fusion_init=None, a_fusion_init=None):
        super(ActionConditionedLSTM, self).__init__(
            in_size, out_size, lateral_init,
            upward_init, bias_init, forget_bias_init)

        if out_size is None:
            out_size, in_size = in_size, None
        self.h_fusion_init = h_fusion_init
        self.a_fusion_init = a_fusion_init

        with self.init_scope():
            self.Wh = linear.Linear(
                out_size, out_size, initialW=0, nobias=True)
            self.Wa = linear.Linear(None, out_size, initialW=0, nobias=True)

    def _initialize_fusion_params(self):
        h_fusion_init = initializers._get_initializer(self.h_fusion_init)
        a_fusion_init = initializers._get_initializer(self.a_fusion_init)

        h_fusion_init(self.Wh.W.array)
        a_fusion_init(self.Wa.W.array)

    def __call__(self, x):
        x, a = x

        if self.upward.W.data is None:
            with cuda.get_device_from_id(self._device_id):
                in_size = functools.reduce(operator.mul, x.shape[1:], 1)
                self.upward._initialize_params(in_size)
                self._initialize_params()

        if self.Wa.W.array is None:
            in_size = a.size // a.shape[0]
            with cuda.get_device_from_id(self._device_id):
                self.Wa._initialize_params(in_size)
                self._initialize_fusion_params()

        batch = x.shape[0]
        lstm_in = self.upward(x)
        if self.h is not None:
            h_size = self.h.shape[0]
            if batch == h_size:
               vt = self.Wh(self.h) * self.Wa(a)
               lstm_in += self.lateral(vt)
            else:
                raise NotImplementedError()
        if self.c is None:
            xp = self.xp
            with cuda.get_device_from_id(self._device_id):
                self.c = variable.Variable(
                    xp.zeros((batch, self.state_size), dtype=x.dtype))
        self.c, y = lstm_activation.lstm(self.c, lstm_in)
        self.h = y
        return y
