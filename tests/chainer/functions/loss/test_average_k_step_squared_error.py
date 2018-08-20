import numpy as np

import unittest
import researchutils.chainer.functions as F


class TestAverageKStepSquaredError(unittest.TestCase):
    def test_average_k_step_squared_error(self):
        k_step = 5
        x1 = np.ones(shape=(3, 3), dtype=np.float32)
        x2 = np.ones(shape=(3, 3), dtype=np.float32)
        actual = F.average_k_step_squared_error(x1, x2, k_step)
        expected = 0

        self.assertEqual(expected, actual.data)

        x1 = np.array([[11, 11, 11],
                       [11, 11, 11],
                       [11, 11, 11]], dtype=np.float32)
        x2 = np.ones(shape=(3, 3), dtype=np.float32)
        actual = F.average_k_step_squared_error(x1, x2, k_step)
        expected = 900 / 5 / 2

        self.assertEqual(expected, actual.data)


if __name__ == '__main__':
    unittest.main()
