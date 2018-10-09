import pytest 
import random
import math
import numpy as np
from researchutils.math.normalization import normalize_by_min_max, standardize

class Testnormalization(object):
    def test_normalize_by_min_max_scalar(self):
        data = np.random.rand(100) * 3.0 + 2.1
        normalized_data = normalize_by_min_max(data)

        assert data.shape == normalized_data.shape
        assert np.argmax(data) == np.argmax(normalized_data)
        assert np.argmin(data) == np.argmin(normalized_data)

        for i in range(len(data)):
            if normalized_data[i] < 0.0 or normalized_data[i] > 1.0:
                assert False        
    
    def test_normalize_by_min_max_matrix(self):
        data = np.random.rand(3, 100) * 3.0 - 2.1
        normalized_data = normalize_by_min_max(data)

        assert data.shape == normalized_data.shape
        assert np.argmin(data) == np.argmin(normalized_data)
        assert np.argmax(data) == np.argmax(normalized_data)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if normalized_data[i, j] < 0.0 or normalized_data[i, j] > 1.0:
                    assert False

    def test_normalize_by_min_max_diff_range(self):
        data = np.random.rand(3, 100) * 3.0 - 2.1
        normalized_data = normalize_by_min_max(data, min_value=-1.0, max_value=5.0)

        assert data.shape == normalized_data.shape
        assert np.argmin(data) == np.argmin(normalized_data)
        assert np.argmax(data) == np.argmax(normalized_data)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if normalized_data[i, j] < -1.0 or normalized_data[i, j] > 5.0:
                    assert False

    def test_normalize_by_min_max_wrong_value(self):
        with pytest.raises(ValueError):
            data = np.random.rand(3, 100) * 3.0 - 2.1
            normalized_data = normalize_by_min_max(data, min_value=1.0, max_value=0.0)
    
    def test_standarized_scalar(self):
        data = np.random.rand(100) * (-5.0) + 1.1
        standarized_data = standardize(data)

        assert data.shape == standarized_data.shape
        assert np.argmin(data) == np.argmin(standarized_data)
        assert np.argmax(data) == np.argmax(standarized_data)
        assert round(np.average(standarized_data), 5) == 0.0
        assert round(np.sqrt(np.var(standarized_data)), 5) == 1.0 

    def test_standarized_matrix(self):
        data = np.random.rand(3, 100) * (-5.0) + 1.1
        standarized_data = standardize(data)

        assert data.shape == standarized_data.shape
        assert np.argmin(data) == np.argmin(standarized_data)
        assert np.argmax(data) == np.argmax(standarized_data)
        assert round(np.average(standarized_data), 5) == 0.0
        assert round(np.sqrt(np.var(standarized_data)), 5) == 1.0

if __name__ == '__main__':
    pytest.main()