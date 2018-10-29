import pytest 
import random
import math
import numpy as np

from researchutils.math.normalization import normalize_by_min_max, standardize

class TestNormalization(object):
    def test_normalize_by_min_max_scalar(self):
        data = np.random.randn(100)
        normalized_data = normalize_by_min_max(data)

        assert data.shape == normalized_data.shape
        assert np.argmax(data) == np.argmax(normalized_data)
        assert np.argmin(data) == np.argmin(normalized_data)

        for i in range(len(data)):
            assert 0.0 <= normalized_data[i] and normalized_data[i] <= 1.0     
    
    def test_normalize_by_min_max_matrix(self):
        data = np.random.randn(2, 100)
        normalized_data = normalize_by_min_max(data)

        assert data.shape == normalized_data.shape
        assert np.argmin(data) == np.argmin(normalized_data)
        assert np.argmax(data) == np.argmax(normalized_data)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                assert 0.0 <= normalized_data[i, j] and normalized_data[i, j] <= 1.0
                    
    def test_normalize_by_min_max_diff_range(self):
        data = np.random.randn(2, 100)
        normalized_data = normalize_by_min_max(data, min_value=-1.0, max_value=5.0)

        assert data.shape == normalized_data.shape
        assert np.argmin(data) == np.argmin(normalized_data)
        assert np.argmax(data) == np.argmax(normalized_data)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                assert -1.0 <= normalized_data[i, j] and normalized_data[i, j] <= 5.0

    def test_normalize_by_min_max_wrong_value(self):
        with pytest.raises(ValueError):
            data = np.random.randn(2, 100)
            normalized_data = normalize_by_min_max(data, min_value=1.0, max_value=0.0)
    
    def test_standarized_scalar(self):
        test_ave = np.random.randint(-100, 100)
        test_dev = np.random.randint(100)
        data = np.random.normal(test_ave, test_dev, size=(100))
        standarized_data = standardize(data)

        assert data.shape == standarized_data.shape
        assert np.argmin(data) == np.argmin(standarized_data)
        assert np.argmax(data) == np.argmax(standarized_data)
        assert np.average(standarized_data) == pytest.approx(0.0)
        assert np.sqrt(np.var(standarized_data)) == pytest.approx(1.0) 

    def test_standarized_matrix(self):
        test_ave = np.random.randint(-100, 100)
        test_dev = np.random.randint(100)
        data = np.random.normal(test_ave, test_dev, size=(2, 100))
        standarized_data = standardize(data)

        assert data.shape == standarized_data.shape
        assert np.argmin(data) == np.argmin(standarized_data)
        assert np.argmax(data) == np.argmax(standarized_data)
        assert np.average(standarized_data) == pytest.approx(0.0)
        assert np.sqrt(np.var(standarized_data)) == pytest.approx(1.0)

if __name__ == '__main__':
    pytest.main()