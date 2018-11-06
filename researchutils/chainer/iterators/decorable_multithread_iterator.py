from chainer.iterators.multithread_iterator import MultithreadIterator
from researchutils.chainer.iterators.unmodifiable_decorable_list import UnmodifiableDecorableList


class DecorableMultithreadIterator(MultithreadIterator):
    """
    MultithreadIterator which enables to configure dataset's end index
    and preprocess dataset for given index before adding to batch

    Preprocess procedure will be done in parallel with multi thread
    to preload batch
    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True,
                 n_threads=1, decor_fun=None, end_index=None):
        decorable_dataset = UnmodifiableDecorableList(
            dataset, decor_fun=decor_fun, end_index=end_index)
        super(DecorableMultithreadIterator, self).__init__(
            dataset=decorable_dataset, batch_size=batch_size, repeat=repeat, shuffle=shuffle, n_threads=n_threads)
