import pytest
import numpy

import chainer
from chainer.backends import cuda
from chainer import functions

from researchutils.chainer import links


class TestGradClipLSTM(object):

    """ GradClipLSTM link test.
    These test were almost same as original chainer's links.LSTM tests
    """

    def setup_method(self, method):
        self.in_size = 10
        self.out_size = 40
        self.link = links.GradClipLSTM(self.in_size, self.out_size)

        self.link.cleargrads()
        x1_shape = (4, self.in_size)
        self.x1 = numpy.random.uniform(-1, 1, x1_shape).astype(numpy.float32)
        x2_shape = (3, self.in_size)
        self.x2 = numpy.random.uniform(-1, 1, x2_shape).astype(numpy.float32)
        x3_shape = (0, self.in_size)
        self.x3 = numpy.random.uniform(-1, 1, x3_shape).astype(numpy.float32)

        self.input_variable = True

    def test_forward(self):
        self.check_forward(self.x1, self.x2, self.x3)

    def check_forward(self, x1_data, x2_data, x3_data):
        xp = self.link.xp
        x1 = chainer.Variable(x1_data) if self.input_variable else x1_data
        h1 = self.link(x1)
        with cuda.get_device_from_array(x1_data):
            c0 = chainer.Variable(xp.zeros((len(self.x1), self.out_size),
                                           dtype=self.x1.dtype))
            c1_expect, h1_expect = functions.lstm(c0, self.link.upward(x1))
        assert h1.data == pytest.approx(h1_expect.data)
        assert self.link.h.data == pytest.approx(h1_expect.data)
        assert self.link.c.data == pytest.approx(c1_expect.data)

        batch = len(x2_data)
        x2 = chainer.Variable(x2_data) if self.input_variable else x2_data
        h1_in, h1_rest = functions.split_axis(
            self.link.h.data, [batch], axis=0)
        y2 = self.link(x2)
        with cuda.get_device_from_array(x1):
            c2_expect, y2_expect = \
                functions.lstm(c1_expect,
                               self.link.upward(x2) + self.link.lateral(h1_in))
        assert y2.data == pytest.approx(y2_expect.data)
        assert self.link.h.data[:batch] == pytest.approx(y2_expect.data)
        assert self.link.h.data[batch:] == pytest.approx(h1_rest.data)

        x3 = chainer.Variable(x3_data) if self.input_variable else x3_data
        h2_rest = self.link.h
        y3 = self.link(x3)
        _, y3_expect = \
            functions.lstm(c2_expect, self.link.upward(x3))
        assert y3.data == pytest.approx(y3_expect.data)
        assert self.link.h.data == pytest.approx(h2_rest.data)


if __name__ == '__main__':
    pytest.main()
