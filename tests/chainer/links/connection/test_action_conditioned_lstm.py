import pytest
import numpy

import chainer
from chainer import initializers
from chainer.backends import cuda
from chainer import functions
from researchutils.chainer import links


class TestActionConditionedLSTM(object):

    """ ActionConditionedLSTM link test.
    These test were almost same as original chainer's links.LSTM tests
    """

    def setup_method(self, method):
        self.in_size = 10
        self.out_size = 40
        self.action_size = 4
        self.link = links.ActionConditionedLSTM(self.in_size, self.out_size)

        self.link.cleargrads()
        x1_shape = (4, self.in_size)
        self.x1 = numpy.random.uniform(-1, 1, x1_shape).astype(numpy.float32)
        a1_shape = (1, self.action_size)
        self.a1 = numpy.random.uniform(-1, 1, a1_shape).astype(numpy.float32)

        x2_shape = (4, self.in_size)
        self.x2 = numpy.random.uniform(-1, 1, x2_shape).astype(numpy.float32)
        a2_shape = (1, self.action_size)
        self.a2 = numpy.random.uniform(-1, 1, a2_shape).astype(numpy.float32)

    def test_forward(self):
        self.check_forward(self.x1, self.a1, self.x2, self.a2)

    def check_forward(self, x1_data, a1_data, x2_data, a2_data):
        xp = self.link.xp
        x1 = chainer.Variable(x1_data)
        a1 = chainer.Variable(a1_data)

        h1 = self.link((x1, a1))
        with cuda.get_device_from_array(x1_data):
            c0 = chainer.Variable(xp.zeros((len(self.x1), self.out_size),
                                           dtype=self.x1.dtype))
            c1_expect, h1_expect = functions.lstm(c0, self.link.upward(x1))
        assert h1.data == pytest.approx(h1_expect.data)
        assert self.link.h.data == pytest.approx(h1_expect.data)
        assert self.link.c.data == pytest.approx(c1_expect.data)

        x2 = chainer.Variable(x2_data)
        a2 = chainer.Variable(a2_data)

        h2 = self.link((x2, a2))
        with cuda.get_device_from_array(x2_data):
            c2_expect, h2_expect = functions.lstm(c1_expect,
                                                  self.link.upward(x2) +
                                                  self.link.lateral(self.link.Wh(h1_expect) * self.link.Wa(a2)))
        assert h2.data == pytest.approx(h2_expect.data)
        assert self.link.h.data == pytest.approx(h2_expect.data)
        assert self.link.c.data == pytest.approx(c2_expect.data)


if __name__ == '__main__':
    pytest.main()
