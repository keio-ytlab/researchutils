import pytest
import numpy as np
import math
import random

from researchutils.math.angle import fit_angle_in_range

def test_fit_angle_in_range():
    angles = np.random.rand(10) + 10.0
    test_pass = True
    correct_angles = fit_angle_in_range(angles)

    for i, angle in enumerate(correct_angles):
        if angle > 2 * math.pi:
            test_pass = False
            break
        elif angle < 0.0:
            test_pass = False
            break
        elif round(math.cos(float(angle)), 5) != round(math.cos(angles[i]), 5):
            test_pass = False
            break

    assert test_pass == True

    angles = np.random.rand(3, 10) + -10.0
    correct_angles = fit_angle_in_range(angles)

    for j in range(correct_angles.shape[0]):
        for i, angle in enumerate(correct_angles[j]):
            if angle > 2 * math.pi:
                test_pass = False
                break
            elif angle < 0.0:
                test_pass = False
                break
            elif round(math.cos(float(angle)), 5) != round(math.cos(angles[j, i]), 5):
                test_pass = False
                break
    
    assert test_pass == True