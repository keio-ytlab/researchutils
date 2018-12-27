import pytest
import numpy as np

import chainer
import chainer.functions as F

from researchutils.chainer import links


class TestConvLSTM(object):
    """ ConvLSTM link test.
    """

    def test_forward(self):
        x = np.ones(shape=(1, 32, 50, 38), dtype=np.float32)
        out_channels = 16
        shape = x.shape
        target_shape = (shape[0], out_channels, shape[2], shape[3])

        initial_c = chainer.Variable(
            np.ones(target_shape, dtype=np.float32))
        initial_h = chainer.Variable(
            np.ones(target_shape, dtype=np.float32))

        conv_lstm = links.ConvLSTM(
            in_channels=32, out_channels=out_channels, ksize=5)
        conv_lstm.reset_state(initial_c, initial_h)

        test_Wci = chainer.Parameter(np.random.rand(out_channels, shape[2], shape[3]).astype(np.float32))
        test_Wcf = chainer.Parameter(np.random.rand(out_channels, shape[2], shape[3]).astype(np.float32))
        test_Wco = chainer.Parameter(np.random.rand(out_channels, shape[2], shape[3]).astype(np.float32))
        
        conv_lstm.Wci = chainer.Parameter(np.reshape(test_Wci.data, (1, out_channels, shape[2], shape[3])))
        conv_lstm.Wcf = chainer.Parameter(np.reshape(test_Wcf.data, (1, out_channels, shape[2], shape[3])))
        conv_lstm.Wco = chainer.Parameter(np.reshape(test_Wco.data, (1, out_channels, shape[2], shape[3])))

        Wxi = conv_lstm.Wxi
        Wxf = conv_lstm.Wxf
        Wxo = conv_lstm.Wxo
        Wxc = conv_lstm.Wxc
        Whi = conv_lstm.Whi
        Whf = conv_lstm.Whf
        Who = conv_lstm.Who
        Whc = conv_lstm.Whc
        Wci = test_Wci
        Wcf = test_Wcf
        Wco = test_Wco

        it = F.sigmoid(Wxi(x) + Whi(initial_h) + F.scale(initial_c, Wci, axis=1))
        ft = F.sigmoid(Wxf(x) + Whf(initial_h) + F.scale(initial_c, Wcf, axis=1))
        ct = ft * initial_c + it * F.tanh(Wxc(x) + Whc(initial_h))
        ot = F.sigmoid(Wxo(x) + Who(initial_h) + F.scale(ct, Wco, axis=1))
        expected = ot * F.tanh(ct)

        actual = conv_lstm(x)

        assert actual.data == pytest.approx(expected.data)

    def test_internal_parameters_initialized(self):
        test_array = np.ones(shape=(1, 128, 50, 38), dtype=np.float32)

        conv_lstm = links.ConvLSTM(
            in_channels=128, out_channels=64, ksize=5)

        assert conv_lstm._internal_state_initialized() == False
        assert conv_lstm._non_convolutional_weights_initialized() == False

        _ = conv_lstm(test_array)

        assert conv_lstm._internal_state_initialized() == True
        assert conv_lstm._non_convolutional_weights_initialized() == True

    def test_reset_state(self):
        test_array = np.ones(shape=(1, 128, 50, 38), dtype=np.float32)

        conv_lstm = links.ConvLSTM(
            in_channels=128, out_channels=64, ksize=5)

        _ = conv_lstm(test_array)

        conv_lstm.reset_state()
        assert conv_lstm._internal_state_initialized() == False

        _ = conv_lstm(test_array)
        assert conv_lstm._internal_state_initialized() == True

    def test_output_size(self):
        test_array = np.ones(shape=(1, 128, 50, 38), dtype=np.float32)

        conv_lstm = links.ConvLSTM(
            in_channels=128, out_channels=64, ksize=5)
        output = conv_lstm(test_array)

        # Image size should remain same
        assert output.shape == (1, 64, 50, 38)

        conv_lstm = links.ConvLSTM(
            in_channels=64, out_channels=64, ksize=3, stride=2)
        output = conv_lstm(output)

        # Image size should be half of original size
        assert output.shape == (1, 64, 25, 19)

    def test_odd_kernel(self):
        test_array = np.ones(shape=(1, 128, 50, 38), dtype=np.float32)

        conv_lstm = links.ConvLSTM(
            in_channels=128, out_channels=64, ksize=5)
        output = conv_lstm(test_array)

        # Image size should remain same
        assert output.shape == (1, 64, 50, 38)

    def test_even_kernel(self):
        with pytest.raises(ValueError):
            _ = links.ConvLSTM(
                in_channels=64, out_channels=64, ksize=4)


if __name__ == '__main__':
    pytest.main()
