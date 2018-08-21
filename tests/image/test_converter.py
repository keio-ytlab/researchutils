import pytest
import numpy as np

from researchutils.image import converter

class TestConverter(object):
    def test_hwc2chw(self):
        image = np.ndarray(shape=(28, 28, 3), dtype=np.float32)
        converted = converter.hwc2chw(image)

        assert converted.shape == (3, 28, 28)

    def test_chw2hwc(self):
        image = np.ndarray(shape=(3, 28, 28), dtype=np.float32)
        converted = converter.chw2hwc(image)

        assert converted.shape == (28, 28, 3)


if __name__ == '__main__':
    pytest.main()
