from chainer.iterators.multiprocess_iterator import MultiprocessIterator
from chainer.iterators.multiprocess_iterator import _PrefetchLoop
from chainer.iterators.multiprocess_iterator import _Communicator
from chainer.iterators.multiprocess_iterator import _PrefetchState
from chainer.iterators.multiprocess_iterator import _raise_timeout_warning
from chainer.iterators.multiprocess_iterator import _measure
from chainer.iterators.multiprocess_iterator import _report_pid
from chainer.iterators.multiprocess_iterator import _pack
from chainer.iterators.multiprocess_iterator import _unpack
from chainer.iterators.multiprocess_iterator import _response_time

import threading
import multiprocessing
import numpy
import signal
import six
import sys


class DecorableMultiprocessIterator(MultiprocessIterator):
    """
    MultiprocessIterator which enables to configure dataset's end index
    and preprocess dataset for given index before adding to batch

    Preprocess procedure will be done in parallel with multi process
    to preload batch

    NOTE: This is an experimental implementation
    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=None,
                 n_processes=None, n_prefetch=1, shared_mem=None, order_sampler=None,
                 dataset_timeout=30.0, decor_fun=None, end_index=None):
        self.end_index = end_index if end_index else len(dataset)
        super(DecorableMultiprocessIterator, self).__init__(
            dataset=dataset, batch_size=batch_size, repeat=repeat, shuffle=shuffle, n_processes=n_processes,
            n_prefetch=n_prefetch, shared_mem=shared_mem, order_sampler=order_sampler, dataset_timeout=dataset_timeout)

        # Override prefetch loop
        self._prefetch_loop = _DecorablePrefetchLoop(self.dataset, self.batch_size, self.repeat,
                                                     self.n_processes, self.n_prefetch, self.shared_mem,
                                                     self._comm, self.order_sampler, self._interruption_testing, decor_fun=decor_fun, end_index=self.end_index)

    def _reset(self):
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.
        if self.order_sampler:
            self._order = self.order_sampler(
                numpy.arange(self.end_index), 0)
        else:
            self._order = None

    @property
    def _epoch_size(self):
        if self._order is None:
            return self.end_index
        else:
            return len(self._order)


class _DecorablePrefetchLoop(_PrefetchLoop):
    def __init__(self, dataset, batch_size, repeat, n_processes, n_prefetch, mem_size, comm, order_sampler, _interruption_testing, decor_fun, end_index):
        self.decor_fun = decor_fun
        self.end_index = end_index
        super(_DecorablePrefetchLoop, self).__init__(dataset=dataset, batch_size=batch_size,
                                                     repeat=repeat, n_processes=n_processes,
                                                     n_prefetch=n_prefetch, mem_size=mem_size, comm=comm,
                                                     order_sampler=order_sampler, _interruption_testing=_interruption_testing)

    def measure(self, dataset_timeout):
        # dataset_timeout: timeout in seconds or None

        status, prefetch_state, _ = self._comm.check()
        if status == _Communicator.STATUS_RESET:
            self.prefetch_state = prefetch_state

        indices = self._proceed()
        if indices is None:  # stop iteration
            batch = None
        else:
            batch_ret = [None]

            def fetch_batch():
                batch_ret[0] = [self.decorate_data(
                    self.dataset, idx) for idx in indices]

            if dataset_timeout is None:
                # Timeout is not set: fetch synchronously
                fetch_batch()
            else:
                # Timeout is set: fetch asynchronously and watch for timeout
                thr = threading.Thread(target=fetch_batch)
                thr.daemon = True
                thr.start()
                thr.join(dataset_timeout)
                if thr.is_alive():
                    _raise_timeout_warning()
                thr.join()

            batch = batch_ret[0]
            self.mem_size = max(map(_measure, batch))
            self._allocate_shared_memory()

        return batch, self.prefetch_state

    def _proceed(self):
        (pos, epoch, is_new_epoch,
            previous_epoch_detail, order) = self.prefetch_state
        n = len(order) if order is not None else self.end_index

        if pos < self.batch_size and epoch > 0 and not self.repeat:
            return None  # stop iteration

        previous_epoch_detail = epoch + pos / n

        new_pos = pos + self.batch_size
        if new_pos < n:
            if order is None:
                indices = numpy.arange(pos, new_pos)
            else:
                indices = order[pos:new_pos]
            is_new_epoch = False
        else:
            new_pos = new_pos - n if self.repeat else 0

            if order is None:
                indices = numpy.arange(pos, n)
                if self.repeat:
                    indices = \
                        numpy.concatenate((indices, numpy.arange(new_pos)))
            else:
                indices = order[pos:n]
                if self.repeat:
                    new_order = self.order_sampler(order, pos)
                    if len(new_order) != n:
                        raise ValueError('The size of order does not match '
                                         'the size of the previous order.')
                    order = new_order
                    indices = \
                        numpy.concatenate((indices, order[:new_pos]))
            epoch += 1
            is_new_epoch = True

        self.prefetch_state = _PrefetchState(
            new_pos, epoch, is_new_epoch,
            previous_epoch_detail, order)
        return indices

    def _task(self):
        # Do a single task in the prefetch thread.
        # Returns a bool indicating whether the loop should continue running.

        status, prefetch_state, reset_count = self._comm.check()
        if status == _Communicator.STATUS_RESET:
            self.prefetch_state = prefetch_state
        elif status == _Communicator.STATUS_TERMINATE:
            return False  # stop loop

        indices = self._proceed()
        if indices is None:  # stop iteration
            batch = None
        else:
            future = self._pool.map_async(_fetch_run, enumerate(indices))
            while True:
                try:
                    data_all = future.get(_response_time)
                except multiprocessing.TimeoutError:
                    if self._comm.is_terminated:
                        return False
                else:
                    break

            batch = [_unpack(data, self.mem_bulk) for data in data_all]

        self._comm.put(batch, self.prefetch_state, reset_count)
        return True

    def launch_thread(self):
        self._pool = multiprocessing.Pool(
            processes=self.n_processes,
            initializer=_fetch_setup,
            initargs=(self.dataset, self.mem_size, self.mem_bulk, self.decorate_data))
        if self._interruption_testing:
            pids = self._pool.map(_report_pid, range(self.n_processes))
            print(' '.join(map(str, pids)))
            sys.stdout.flush()

        thread = threading.Thread(target=self._run, name='prefetch_loop')
        thread.setDaemon(True)
        thread.start()
        self._thread = thread
        return thread

    def decorate_data(self, dataset, index):
        if self.decor_fun:
            return self.decor_fun(dataset, index)
        return dataset[index]


# Overriding from MultiprocessIterator
_fetch_dataset = None
_fetch_mem_size = None
_fetch_mem_bulk = None
_fetch_decor_fun = None


def _fetch_setup(dataset, mem_size, mem_bulk, decor_fun):
    global _fetch_dataset, _fetch_mem_size, _fetch_mem_bulk, _fetch_decor_fun
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _fetch_dataset = dataset
    _fetch_mem_size = mem_size
    _fetch_mem_bulk = mem_bulk
    _fetch_decor_fun = decor_fun


def _fetch_run(inputs):
    i, index = inputs
    data = _fetch_decor_fun(_fetch_dataset, index)
    if _fetch_mem_bulk is not None:
        offset = i * _fetch_mem_size
        limit = offset + _fetch_mem_size
        data = _pack(data, _fetch_mem_bulk, offset, limit)
    return data
