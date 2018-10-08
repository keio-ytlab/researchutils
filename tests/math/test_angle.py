import pytest
import numpy as np
import math
import random

from researchutils.math.angle import fit_angle_in_rad_range, fit_angle_in_deg_range

class TestAngle(object):
    def test_fit_angle_in_rad_range_scalar(self):
        angles = np.random.rand(10) + 10.0
        correct_angles = fit_angle_in_rad_range(angles)

        for i, angle in enumerate(correct_angles):
            if angle > 2 * math.pi:
                assert False
            elif angle < 0.0:
                assert False
            elif round(math.cos(float(angle)), 5) != round(math.cos(angles[i]), 5):
                assert False
            elif round(math.sin(float(angle)), 5) != round(math.sin(angles[i]), 5):
                assert False

    def test_fit_angle_in_rad_range_matrix(self):
        angles = np.random.rand(3, 10) + -10.0
        correct_angles = fit_angle_in_rad_range(angles)

        for i in range(correct_angles.shape[0]):
            for j, angle in enumerate(correct_angles[i]):
                if angle > 2 * math.pi:
                    assert False
                elif angle < 0.0:
                    assert False
                elif round(math.cos(float(angle)), 5) != round(math.cos(angles[i, j]), 5):
                    assert False
                elif round(math.sin(float(angle)), 5) != round(math.sin(angles[i, j]), 5):
                    assert False

    def test_fit_angle_in_diff_rad_range_matrix(self):
        angles = np.random.rand(3, 10) + -10.0
        correct_angles = fit_angle_in_rad_range(angles, max_angle=math.pi, min_angle=-math.pi)

        for i in range(correct_angles.shape[0]):
            for j, angle in enumerate(correct_angles[i]):
                if angle > math.pi:
                    assert False
                elif angle < -math.pi:
                    assert False
                elif round(math.cos(float(angle)), 5) != round(math.cos(angles[i, j]), 5):
                    assert False
                elif round(math.sin(float(angle)), 5) != round(math.sin(angles[i, j]), 5):
                    assert False
    
    def test_fit_angle_in_rad_range_wrong_range(self):
        with pytest.raises(ValueError):
            angles = np.random.rand(10)
            fit_angle_in_range(angles, min_angle=0.0, max_angle=math.pi)

    def test_fit_angle_in_rad_range_min_is_greater_than_max(self):
        with pytest.raises(ValueError):
            angles = np.random.rand(10)
            fit_angle_in_range(angles, min_angle=math.pi, max_angle=-math.pi)

    def test_fit_angle_in_deg_range_scalar(self):
        angles = np.random.rand(10) + random.choice([-500.0, 500.0])
        correct_angles = fit_angle_in_deg_range(angles)

        for i, angle in enumerate(correct_angles):
            if angle < 0.0 or angle > 360.0:
                assert False

            if round(math.cos(float(math.radians(angle))), 5) != round(math.cos(math.radians(angles[i])), 5) or\
               round(math.sin(float(math.radians(angle))), 5) != round(math.sin(math.radians(angles[i])), 5):
                assert False

    def test_fit_angle_in_deg_range_matrix(self):
        angles = np.random.rand(3, 10) + random.choice([-500.0, 500.0])
        correct_angles = fit_angle_in_deg_range(angles)

        for i in range(correct_angles.shape[0]):
            for j, angle in enumerate(correct_angles[i]):
                if angle < 0.0 or angle > 360.0:
                    assert False

                if round(math.cos(float(math.radians(angle))), 5) != round(math.cos(math.radians(angles[i, j])), 5) or\
                   round(math.sin(float(math.radians(angle))), 5) != round(math.sin(math.radians(angles[i, j])), 5):
                    assert False

    def test_fit_angle_in_diff_deg_range_matrix(self):
        angles = np.random.rand(3, 10) + random.choice([-500.0, 500.0])
        correct_angles = fit_angle_in_deg_range(angles, max_angle=180.0, min_angle=-180.0)

        for i in range(correct_angles.shape[0]):
            for j, angle in enumerate(correct_angles[i]):
                if angle < -180.0 or angle > 180.0:
                    assert False

                if round(math.cos(float(math.radians(angle))), 5) != round(math.cos(math.radians(angles[i, j])), 5) or\
                   round(math.sin(float(math.radians(angle))), 5) != round(math.sin(math.radians(angles[i, j])), 5):
                    assert False
    
    def test_fit_angle_in_deg_range_wrong_range(self):
        with pytest.raises(ValueError):
            angles = np.random.rand(10)
            fit_angle_in_deg_range(angles, min_angle=180.0, max_angle=0.0)

    def test_fit_angle_in_deg_range_min_is_greater_than_max(self):
        with pytest.raises(ValueError):
            angles = np.random.rand(10)
            fit_angle_in_deg_range(angles, min_angle=180.0, max_angle=-180.0)
    

if __name__=="__main__":
    pytest.main()