import numpy as np
import pytest
import mock
from researchutils.image import preprocess


class TestPreprocess(object):
    def test_compute_mean_image(self):
        image_shape = (28, 28, 3)
        images = [np.ones(shape=image_shape, dtype=np.uint8)
                  for i in range(10)]
        mean_image = preprocess.compute_mean_image(images)

        assert mean_image.shape == image_shape
        assert mean_image.dtype == np.float32
        assert mean_image[0][0][0] == 1
        assert mean_image[0][0][1] == 1
        assert mean_image[0][0][2] == 1

    def test_compute_mean_image_with_difference_size(self):
        image_shape = (28, 28, 3)
        images = [np.ones(shape=image_shape, dtype=np.uint8)
                  for i in range(10)]

        irregular_image = np.ones(shape=(100, 100, 3), dtype=np.uint8)
        images.append(irregular_image)

        with pytest.raises(ValueError):
            preprocess.compute_mean_image(images)


if __name__ == '__main__':
    pytest.main()
