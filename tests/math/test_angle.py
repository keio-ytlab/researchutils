import pytest
import numpy as np
import math
import random

from researchutils.math.angle import fit_angle_in_range

def test_angle_range_correcter():
    angles = np.random.rand(10) + 10.0
    test_pass = True
    correct_angles = fit_angle_in_range(angles)

    for i, angle in enumerate(correct_angles):
        if angle > math.pi:
            test_pass = False
            break
        elif angle < -math.pi:
            test_pass = False
            break
        elif round(math.cos(float(angle)), 5) != round(math.cos(angles[i]), 5):
            test_pass = False
            break

    assert test_pass == True

    angles = np.random.rand(2, 10) + -10.0
    correct_angles = fit_angle_in_range(angles)

    for i, angle in enumerate(correct_angles[0]):
        if angle > math.pi:
            test_pass = False
            break
        elif angle < -math.pi:
            test_pass = False
            break
        elif round(math.cos(float(angle)), 5) != round(math.cos(angles[0, i]), 5):
            test_pass = False
            break
    
    assert test_pass == True