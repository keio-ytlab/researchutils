from builtins import range


class UnmodifiableDecorableList(list):
    """
    Unmodifiable list which can decorate items in the list by providing decor_fun

    Parameters
    -------
    items : iterable
        items of the list
    decor_fun : callable or None
        function to apply everytime __getitem__ is called
    end_index : integer or None
        length of the list (exclusive) to announce to the user of this list
        can be same or smaller than the length of the given items.
        if end_index is None, then the length of this list will be same as length of given items

    """

    def __init__(self, items, decor_fun=None, end_index=None):
        super(UnmodifiableDecorableList, self).__init__(items)
        self.decor_fun = decor_fun
        self.end_index = end_index if end_index else len(items)
        self.items = items

    def __len__(self):
        return self.end_index

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop < self.end_index else self.end_index
            step = key.step or 1
            return [self._get_decorated_data(self.items, index) for index in range(start, stop, step)]
        else:
            return self._get_decorated_data(self.items, key)

    # For python 2 compatibility
    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def insert(self, item):
        raise NotImplementedError('This list is unmodifiable!!')

    def remove(self, item):
        raise NotImplementedError('This list is unmodifiable!!')

    def append(self, item):
        raise NotImplementedError('This list is unmodifiable!!')

    def extend(self, items):
        raise NotImplementedError('This list is unmodifiable!!')

    def sort(self, key=None, reverse=False):
        raise NotImplementedError('This list is unmodifiable!!')

    def pop(self, index):
        raise NotImplementedError('This list is unmodifiable!!')

    def _get_decorated_data(self, dataset, index):
        if self.decor_fun is None:
            return dataset[index]
        else:
            return self.decor_fun(dataset, index)
