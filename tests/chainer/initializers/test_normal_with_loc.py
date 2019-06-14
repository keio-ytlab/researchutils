import numpy as np

import pytest
from researchutils.chainer.initializers import normal_with_loc


class TestNormalWithLoc(object):
    def test_normal_with_loc(self):
        mean = 5.0
        var = 1.0
        initializer = normal_with_loc.NormalWithLoc(loc=mean, scale=var)

        shape = (10000, 1)
        target_array = np.ndarray(shape=shape, dtype=np.float32)
        initializer(target_array)

        computed_mean = np.mean(target_array)
        computed_var = np.var(target_array)

        print('mean: ', computed_mean, 'var:', computed_var)

        assert computed_mean == pytest.approx(mean, rel=1e-1)
        assert computed_var == pytest.approx(var, rel=1e-1)


if __name__ == '__main__':
    pytest.main()
