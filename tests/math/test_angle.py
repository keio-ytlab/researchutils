import pytest
import numpy as np
import math
import random

from researchutils.math.angle import fit_angle_in_range

class TestAngle(object):
    def test_fit_angle_in_range_scalar(self):
        angles = np.random.rand(10) + 10.0
        correct_angles = fit_angle_in_range(angles)

        for i, angle in enumerate(correct_angles):
            if angle > 2 * math.pi:
                assert False
            elif angle < 0.0:
                assert False
            elif round(math.cos(float(angle)), 5) != round(math.cos(angles[i]), 5):
                assert False
            elif round(math.sin(float(angle)), 5) != round(math.sin(angles[i]), 5):
                assert False

    def test_fit_angle_in_range_matrix(self):
        angles = np.random.rand(3, 10) + -10.0
        correct_angles = fit_angle_in_range(angles)

        for j in range(correct_angles.shape[0]):
            for i, angle in enumerate(correct_angles[j]):
                if angle > 2 * math.pi:
                    assert False
                elif angle < 0.0:
                    assert False
                elif round(math.cos(float(angle)), 5) != round(math.cos(angles[j, i]), 5):
                    assert False
                elif round(math.sin(float(angle)), 5) != round(math.sin(angles[j, i]), 5):
                    assert False

    def test_fit_angle_in_diff_range_matrix(self):
        angles = np.random.rand(3, 10) + -10.0
        correct_angles = fit_angle_in_range(angles, max_angle=math.pi, min_angle=-math.pi)

        for j in range(correct_angles.shape[0]):
            for i, angle in enumerate(correct_angles[j]):
                if angle > math.pi:
                    assert False
                elif angle < -math.pi:
                    assert False
                elif round(math.cos(float(angle)), 5) != round(math.cos(angles[j, i]), 5):
                    assert False
                elif round(math.sin(float(angle)), 5) != round(math.sin(angles[j, i]), 5):
                    assert False
    
    def test_fit_angle_in_range_wrong_range(self):
        with pytest.raises(ValueError):
            angles = np.random.rand(10)
            fit_angle_in_range(angles, min_angle=0.0, max_angle=math.pi)

    def test_fit_angle_in_range_min_is_greater_than_max(self):
        with pytest.raises(ValueError):
            angles = np.random.rand(10)
            fit_angle_in_range(angles, min_angle=math.pi, max_angle=-math.pi)

if __name__=="__main__":
    pytest.main()