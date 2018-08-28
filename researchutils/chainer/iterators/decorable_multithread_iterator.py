from multiprocessing import pool
from chainer.iterators.multithread_iterator import MultithreadIterator

import numpy
import six

class DecorableMultithreadIterator(MultithreadIterator):
    """
    MultithreadIterator which enables to configure dataset's end index
    and preprocess dataset for given index before adding to batch

    Preprocess procedure will be done in parallel with multi thread
    to preload batch
    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True,
                 n_threads=1, decor_fun=None, end_index=None):
        self.decor_fun = decor_fun
        self.end_index = end_index if end_index else len(dataset)
        super(DecorableMultithreadIterator, self).__init__(
            dataset=dataset, batch_size=batch_size, repeat=repeat, shuffle=shuffle, n_threads=n_threads)

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self.end_index

    def _read(self, args):
        dataset, index = args
        if self.decor_fun:
            return self.decor_fun(dataset, index)
        return dataset[index]

    def _invoke_prefetch(self):
        assert self._next is None
        if not self._repeat and self.epoch > 0:
            return
        if self._pool is None:
            self._pool = pool.ThreadPool(self.n_threads)
        n = self.end_index
        i = self.current_position

        order = self._order
        args = []
        dataset = self.dataset
        epoch = self.epoch
        is_new_epoch = False
        for _ in six.moves.range(self.batch_size):
            index = i if order is None else order[i]
            args.append((dataset, index))
            i += 1
            if i >= n:
                epoch += 1
                is_new_epoch = True
                i = 0
                if not self._repeat:
                    break
                if order is not None:
                    # We cannot shuffle the order directly here, since the
                    # iterator may be serialized before the prefetched data are
                    # consumed by the user, in which case an inconsistency
                    # appears.
                    order = order.copy()
                    numpy.random.shuffle(order)

        self._next = self._pool.map_async(self._read, args)
        self._next_state = (i, epoch, is_new_epoch, order)

    def reset(self):
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        if self._shuffle:
            self._order = numpy.random.permutation(self.end_index)
        else:
            self._order = None

        # reset internal state
        self._next = None
        self._previous_epoch_detail = None
