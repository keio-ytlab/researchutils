import pytest

import numpy

import chainer
from chainer import functions
from chainer import gradient_check

from researchutils.chainer.functions.activation import grad_clip_lstm


def _sigmoid(x):
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half


class TestGradClipLSTM(object):
    def setup_method(self, method):
        self.batch = 3
        self.dtype = numpy.float32

        dtype = self.dtype
        hidden_shape = (3, 2, 4)
        x_shape = (self.batch, 8, 4)
        y_shape = (self.batch, 2, 4)

        c_prev = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)

        gc = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        gh = numpy.random.uniform(-1, 1, y_shape).astype(dtype)

        ggc = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        ggx = numpy.random.uniform(-1, 1, x_shape).astype(dtype)

        self.inputs = [c_prev, x]
        self.grad_outputs = [gc, gh]

        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}

    def test_forward(self):
        self.check_forward(self.inputs)

    def forward_cpu(self, inputs):
        c_prev, x = inputs
        batch = x.shape[0]
        a_in = x[:, [0, 4]]
        i_in = x[:, [1, 5]]
        f_in = x[:, [2, 6]]
        o_in = x[:, [3, 7]]
        c_expect = (_sigmoid(i_in) * numpy.tanh(a_in)
                    + _sigmoid(f_in) * c_prev[:batch])
        h_expect = _sigmoid(o_in) * numpy.tanh(c_expect)
        return c_expect, h_expect

    def check_forward(self, inputs):
        # Compute expected out
        c_prev, x = inputs
        batch = x.shape[0]
        c_expect_2 = c_prev[batch:]
        c_expect_1, h_expect = self.forward_cpu(inputs)

        inputs = [chainer.Variable(xx) for xx in inputs]

        c, h = grad_clip_lstm.grad_clip_lstm(*inputs)
        assert c.data.dtype == self.dtype
        assert h.data.dtype == self.dtype

        assert c_expect_1 == pytest.approx(c.data[:batch])
        assert c_expect_2 == pytest.approx(c.data[batch:])
        assert h_expect == pytest.approx(h.data)

    def test_backward(self):
        self.check_backward(self.inputs, self.grad_outputs)

    def test_clipped_backward(self):
        dtype = self.dtype
        hidden_shape = (3, 2, 4)
        x_shape = (self.batch, 8, 4)
        y_shape = (self.batch, 2, 4)

        c_prev = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        gc = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        
        # Give huge input to check effect of clipping
        gh = numpy.random.uniform(-1, 1, y_shape).astype(dtype) * 1000

        self.check_clipped_backward([c_prev, x], [gc, gh])

    def check_backward(self, inputs, grad_outputs):
        gradient_check.check_backward(
            grad_clip_lstm.grad_clip_lstm, inputs, grad_outputs,
            **self.check_backward_options)

    def check_clipped_backward(self, inputs, grad_outputs):
        c_prev_data, x_data = inputs
        c_prev = chainer.Variable(c_prev_data)
        x = chainer.Variable(x_data)

        clip_min = -1.0
        clip_max = 1.0
        c, y = grad_clip_lstm.grad_clip_lstm(c_prev, x, clip_min=clip_min, clip_max=clip_max)
        gc, gy = grad_outputs
        c.grad = gc
        y.grad = gy

        y.backward()

        c_prev.grad
        assert (clip_min <= x.grad).all() and (x.grad <= clip_max).all()


if __name__ == '__main__':
    pytest.main()
