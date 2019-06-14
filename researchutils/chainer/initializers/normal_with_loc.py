import numpy as np

from chainer import backend
from chainer.backends import cuda
from chainer import initializer


class NormalWithLoc(initializer.Initializer):
    def __init__(self, loc=0.0, scale=0.05, dtype=None):
        self.loc = loc
        self.scale = scale
        super(NormalWithLoc, self).__init__(dtype)

    def __call__(self, array):
        xp = backend.get_array_module(array)
        args = {'loc': self.loc, 'scale': self.scale, 'size': array.shape}
        if xp is cuda.cupy:
            if self.dtype == np.float32 or self.dtype == np.float16:
                args['dtype'] = np.float32
        array[...] = xp.random.normal(**args)
