from chainer.iterators.multiprocess_iterator import MultiprocessIterator
from researchutils.chainer.iterators.unmodifiable_decorable_list import UnmodifiableDecorableList


class DecorableMultiprocessIterator(MultiprocessIterator):
    """
    MultiprocessIterator which enables to configure dataset's end index
    and preprocess dataset for given index before adding to batch

    Preprocess procedure will be done in parallel with multi process
    to preload batch

    NOTE: This is an experimental implementation
    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=None,
                 n_processes=None, n_prefetch=1, shared_mem=None, decor_fun=None, end_index=None):
        decorable_dataset = UnmodifiableDecorableList(
            dataset, decor_fun=decor_fun, end_index=end_index)
        super(DecorableMultiprocessIterator, self).__init__(
            dataset=decorable_dataset, batch_size=batch_size, repeat=repeat, shuffle=shuffle, n_processes=n_processes,
            n_prefetch=n_prefetch, shared_mem=shared_mem)
