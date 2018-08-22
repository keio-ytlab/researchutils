import pytest
import numpy as np
from researchutils import arrays


class TestArrays(object):
    def test_one_hot_vector(self):
        one_hot_vector = arrays.one_hot(
            indices=[1], shape=(3, 1), dtype=np.float32)
        print('one_hot_vector: {}'.format(one_hot_vector))
        assert one_hot_vector.dtype == np.float32
        assert one_hot_vector[0] == 0
        assert one_hot_vector[1] == 1

    def test_one_hot_matrix(self):
        one_hot_vector = arrays.one_hot(
            indices=[0, 1, 1], shape=(3, 3), dtype=np.int8)
        print('one_hot_vector: {}'.format(one_hot_vector))
        assert one_hot_vector.dtype == np.int8
        assert one_hot_vector[0][0] == 1
        assert one_hot_vector[0][1] == 0
        assert one_hot_vector[1][1] == 1
        assert one_hot_vector[1][0] == 0
        assert one_hot_vector[1][2] == 1
        assert one_hot_vector[2][0] == 0

    def test_one_hot_no_dtype(self):
        one_hot_vector = arrays.one_hot(indices=[0, 1, 2], shape=(3, 3))
        assert one_hot_vector.dtype == np.int8
        assert one_hot_vector[0][0] == 1
        assert one_hot_vector[1][1] == 1
        assert one_hot_vector[2][2] == 1

    def test_one_hot_on_value_float(self):
        one_hot_vector = arrays.one_hot(
            indices=[0, 1, 2], shape=(3, 3), on_value=1.0)
        assert one_hot_vector.dtype == np.float32
        assert one_hot_vector[0][0] == 1
        assert one_hot_vector[1][1] == 1
        assert one_hot_vector[2][2] == 1

    def test_one_hot_off_value_float(self):
        one_hot_vector = arrays.one_hot(
            indices=[0, 1, 2], shape=(3, 3), off_value=0.0)
        assert one_hot_vector.dtype == np.float32
        assert one_hot_vector[0][0] == 1
        assert one_hot_vector[1][1] == 1
        assert one_hot_vector[2][2] == 1

    def test_one_hot_3x3_tensor(self):
        with pytest.raises(ValueError):
            arrays.one_hot(indices=range(9), shape=(3, 3, 3), off_value=0.0)


if __name__ == "__main__":
    pytest.main()
