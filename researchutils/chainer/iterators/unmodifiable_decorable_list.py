from builtins import range


class UnmodifiableDecorableList(list):
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

    def insert(self, item):
        raise NotImplementedError('This list is unmodifiable!!')

    def remove(self, item):
        raise NotImplementedError('This list is unmodifiable!!')

    def append(self, item):
        raise NotImplementedError('This list is unmodifiable!!')

    def extend(self, items):
        raise NotImplementedError('This list is unmodifiable!!')

    def sort(self, key, reverse):
        raise NotImplementedError('This list is unmodifiable!!')

    def pop(self, index):
        raise NotImplementedError('This list is unmodifiable!!')

    def _get_decorated_data(self, dataset, index):
        if self.decor_fun is None:
            return dataset[index]
        else:
            return self.decor_fun(dataset, index)
