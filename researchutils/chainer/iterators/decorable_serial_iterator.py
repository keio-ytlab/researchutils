import numpy
from chainer.iterators.serial_iterator import SerialIterator


class DecorableSerialIterator(SerialIterator):
    """
    SerialIterator which enables to configure dataset's end index
    and preprocess dataset for given index before adding to batch

    Preprocess procedure will be done on caller's thread
    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True, decor_fun=None, end_index=None):
        self.decor_fun = decor_fun
        self.end_index = end_index if end_index else len(dataset)
        super(DecorableSerialIterator, self).__init__(
            dataset, batch_size, repeat=repeat, shuffle=shuffle)

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size
        N = self.end_index

        if self._order is None:
            last_index = i_end if i_end < N else N
            batch = [self.decorate_data(self.dataset, index)
                     for index in range(i, last_index)]
        else:
            batch = [self.decorate_data(self.dataset, index)
                     for index in self._order[i:i_end]]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    numpy.random.shuffle(self._order)
                if rest > 0:
                    if self._order is None:
                        batch.extend([self.decorate_data(self.dataset, index)
                                      for index in range(0, rest)])
                    else:
                        batch.extend([self.decorate_data(self.dataset, index)
                                      for index in self._order[:rest]])
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        if not len(batch) == self.batch_size and self.repeat == True:
            raise ValueError('created batch size {} differs from expected batch size {}'.format(
                len(batch), self.batch_size))
        return batch

    next = __next__

    def decorate_data(self, dataset, index):
        if self.decor_fun:
            return self.decor_fun(dataset, index)
        return dataset[index]

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self.end_index

    def reset(self):
        if self._shuffle:
            self._order = numpy.random.permutation(self.end_index)
        else:
            self._order = None

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.