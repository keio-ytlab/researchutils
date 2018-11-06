from chainer.iterators.serial_iterator import SerialIterator
from researchutils.chainer.iterators.unmodifiable_decorable_list import UnmodifiableDecorableList


class DecorableSerialIterator(SerialIterator):
    """
    SerialIterator which enables to configure dataset's end index
    and preprocess dataset for given index before adding to batch

    Preprocess procedure will be done on caller's thread
    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True, decor_fun=None, end_index=None):
        decorable_dataset = UnmodifiableDecorableList(
            dataset, decor_fun=decor_fun, end_index=end_index)
        super(DecorableSerialIterator, self).__init__(
            decorable_dataset, batch_size, repeat=repeat, shuffle=shuffle)
