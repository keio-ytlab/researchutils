import pytest
import numpy as np

import chainer
import chainer.functions as F

from researchutils.chainer import links


class TestDeconvolutionPS(object):
    """ DeconvolutionPS link test.
    """

    def test_output_size(self):
        test_array = np.ones(shape=(1, 128, 11, 8), dtype=np.float32)

        # 11x8 -> 24x18
        ps = links.DeconvolutionPS(
            in_channels=128, out_channels=128, ksize=4, stride=2)
        output = ps(test_array)

        assert output.shape == (1, 128, 24, 18)

        ps = links.DeconvolutionPS(
            in_channels=128, out_channels=128, ksize=6, stride=2, pad=(1, 1))
        output = ps(output)

        assert output.shape == (1, 128, 50, 38)

        ps = links.DeconvolutionPS(
            in_channels=128, out_channels=64, ksize=6, stride=2, pad=(1, 1))
        output = ps(output)

        assert output.shape == (1, 64, 102, 78)

        ps = links.DeconvolutionPS(
            in_channels=64, out_channels=3, ksize=8, stride=2, pad=(0, 1))
        output = ps(output)

        assert output.shape == (1, 3, 210, 160)

    def test_wrong_ksize_and_stride(self):
        with pytest.raises(ValueError):
            links.DeconvolutionPS(
                in_channels=128, out_channels=128, ksize=4, stride=3)


if __name__ == '__main__':
    pytest.main()
