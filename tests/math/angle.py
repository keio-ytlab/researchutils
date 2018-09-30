import pytest
import numpy as np
import math
import os, sys
import random

from researchutils.math.angle import deg_to_rad, rad_to_deg, angle_range_corrector

def test_deg_to_rad():
    assert np.round(deg_to_rad(np.array([90.0])), 5) == np.array([round(math.pi * 0.5, 5)])
    assert round(deg_to_rad(70.0), 5) == 1.22173
    assert round(deg_to_rad(240.0), 5) == 4.18879
    assert round(deg_to_rad(70), 5) == 1.22173

def test_rad_to_deg():
    assert np.round(rad_to_deg(np.array([math.pi * 0.5])), 5) == np.array([90.0])
    assert round(rad_to_deg(1.22173), 3) == 70.0
    assert round(rad_to_deg(4.18879), 3) == 240.0

def test_angle_range_correcter():
    angles = np.random.rand(10) + 3.0
    test_pass = True
    correct_angles = angle_range_corrector(angles)

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

    angles = np.random.rand(1, 10) + 3.0
    correct_angles = angle_range_corrector(angles)

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