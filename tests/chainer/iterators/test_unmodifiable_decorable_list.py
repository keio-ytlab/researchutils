import pytest
from researchutils.chainer.iterators import unmodifiable_decorable_list


class TestUnmodifiableDecorableList(object):
    def test_len_no_end_index(self):
        dataset = [data for data in range(50)]
        decorable_list = unmodifiable_decorable_list.UnmodifiableDecorableList(
            dataset, decor_fun=None)
        assert len(decorable_list) == len(dataset)

    def test_len_with_end_index(self):
        dataset = [data for data in range(50)]
        fake_end_index = 25
        decorable_list = unmodifiable_decorable_list.UnmodifiableDecorableList(
            dataset, decor_fun=None, end_index=fake_end_index)
        assert len(decorable_list) == fake_end_index

    def test_getitem_index(self):
        def _decor_fun(items, index):
            return items[index] * 2
        dataset = [data for data in range(50)]
        decorable_list = unmodifiable_decorable_list.UnmodifiableDecorableList(
            dataset, decor_fun=_decor_fun)
        assert decorable_list[10] == dataset[10] * 2

    def test_getitem_slice(self):
        def _decor_fun(items, index):
            return items[index] * 2
        dataset = [data for data in range(50)]
        decorable_list = unmodifiable_decorable_list.UnmodifiableDecorableList(
            dataset, decor_fun=_decor_fun)
        start = 20
        end = 30
        sliced = decorable_list[start:end]
        assert len(sliced) == 10
        for index in range(len(sliced)):
            sliced[index] == dataset[start + index] * 2

    def test_getitem_slice_larger_than_end_index(self):
        def _decor_fun(items, index):
            return items[index] * 2
        dataset = [data for data in range(50)]
        end_index = 10
        decorable_list = unmodifiable_decorable_list.UnmodifiableDecorableList(
            dataset, decor_fun=_decor_fun, end_index=end_index)
        start = 5
        end = 30
        sliced = decorable_list[start:end]
        assert len(sliced) == 5
        for index in range(len(sliced)):
            sliced[index] == dataset[start + index] * 2

    def test_insert(self):
        def _decor_fun(items, index):
            return items[index] * 2
        dataset = [data for data in range(50)]
        end_index = 10
        decorable_list = unmodifiable_decorable_list.UnmodifiableDecorableList(
            dataset, decor_fun=_decor_fun, end_index=end_index)

        with pytest.raises(NotImplementedError):
            decorable_list.insert(100)

    def test_remove(self):
        def _decor_fun(items, index):
            return items[index] * 2
        dataset = [data for data in range(50)]
        end_index = 10
        decorable_list = unmodifiable_decorable_list.UnmodifiableDecorableList(
            dataset, decor_fun=_decor_fun, end_index=end_index)

        with pytest.raises(NotImplementedError):
            decorable_list.remove(100)

    def test_append(self):
        def _decor_fun(items, index):
            return items[index] * 2
        dataset = [data for data in range(50)]
        end_index = 10
        decorable_list = unmodifiable_decorable_list.UnmodifiableDecorableList(
            dataset, decor_fun=_decor_fun, end_index=end_index)

        with pytest.raises(NotImplementedError):
            decorable_list.append(100)

    def test_extend(self):
        def _decor_fun(items, index):
            return items[index] * 2
        dataset = [data for data in range(50)]
        end_index = 10
        decorable_list = unmodifiable_decorable_list.UnmodifiableDecorableList(
            dataset, decor_fun=_decor_fun, end_index=end_index)

        with pytest.raises(NotImplementedError):
            decorable_list.extend([100])

    def test_sort(self):
        def _decor_fun(items, index):
            return items[index] * 2
        dataset = [data for data in range(50)]
        end_index = 10
        decorable_list = unmodifiable_decorable_list.UnmodifiableDecorableList(
            dataset, decor_fun=_decor_fun, end_index=end_index)

        with pytest.raises(NotImplementedError):
            decorable_list.sort(key=len, reverse=False)

    def test_pop(self):
        def _decor_fun(items, index):
            return items[index] * 2
        dataset = [data for data in range(50)]
        end_index = 10
        decorable_list = unmodifiable_decorable_list.UnmodifiableDecorableList(
            dataset, decor_fun=_decor_fun, end_index=end_index)

        with pytest.raises(NotImplementedError):
            decorable_list.pop(0)


if __name__ == '__main__':
    pytest.main()
