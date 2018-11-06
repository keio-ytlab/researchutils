import pytest
from researchutils.chainer.iterators import decorable_multiprocess_iterator


class TestDecorableMultiprocessIterator(object):
    def test_next_repeat_no_shuffle(self):
        dataset = [data for data in range(50)]
        batch_size = 16
        iterator = decorable_multiprocess_iterator.DecorableMultiprocessIterator(dataset=dataset,
                                                                                 batch_size=batch_size,
                                                                                 repeat=True,
                                                                                 shuffle=False)
        for times in range(len(dataset) // batch_size):
            batch = iterator.next()
            assert len(batch) == batch_size
            # should be same order as provided dataset
            for index in range(len(batch)):
                assert batch[index] == dataset[times * batch_size + index]

        # Should always contain same batch_size
        for _ in range(10):
            batch = iterator.next()
            assert len(batch) == batch_size

    def test_next_repeat_with_shuffle(self):
        dataset = [data for data in range(50)]
        batch_size = 16
        iterator = decorable_multiprocess_iterator.DecorableMultiprocessIterator(dataset=dataset,
                                                                                 batch_size=batch_size,
                                                                                 repeat=True,
                                                                                 shuffle=True)
        data_map = {}
        for _ in range(len(dataset) // batch_size):
            batch = iterator.next()
            assert len(batch) == batch_size
            # Check batch contains different value of provided dataset
            for index in range(len(batch)):
                data = batch[index]
                assert data in dataset
                assert not (data in data_map)
                data_map[data] = True
        batch = iterator.next()

        # Should always contain same batch_size
        for _ in range(10):
            batch = iterator.next()
            assert len(batch) == batch_size

    def test_next_no_repeat_no_shuffle(self):
        dataset = [data for data in range(50)]
        batch_size = 16
        iterator = decorable_multiprocess_iterator.DecorableMultiprocessIterator(dataset=dataset,
                                                                                 batch_size=batch_size,
                                                                                 repeat=False,
                                                                                 shuffle=False)
        for times in range(len(dataset) // batch_size):
            batch = iterator.next()
            assert len(batch) == batch_size
            # should be same order as provided dataset
            for index in range(len(batch)):
                assert batch[index] == dataset[times * batch_size + index]

        # Last batch should only have 2 items
        batch = iterator.next()
        assert len(batch) == 2

    def test_next_no_repeat_with_shuffle(self):
        dataset = [data for data in range(50)]
        batch_size = 16
        iterator = decorable_multiprocess_iterator.DecorableMultiprocessIterator(dataset=dataset,
                                                                                 batch_size=batch_size,
                                                                                 repeat=False,
                                                                                 shuffle=True)
        data_map = {}
        for _ in range(len(dataset) // batch_size):
            batch = iterator.next()
            assert len(batch) == batch_size
            # Check batch contains different value of provided dataset
            for index in range(len(batch)):
                data = batch[index]
                assert data in dataset
                assert not (data in data_map)
                data_map[data] = True

        # Last batch should only have 2 items
        batch = iterator.next()
        assert len(batch) == 2

    def test_decorate_data(self):
        def decorate_data(dataset, index):
            return dataset[index] + 1
        dataset = [data for data in range(50)]
        batch_size = 16
        iterator = decorable_multiprocess_iterator.DecorableMultiprocessIterator(dataset=dataset,
                                                                                 batch_size=batch_size,
                                                                                 repeat=False,
                                                                                 shuffle=False,
                                                                                 decor_fun=decorate_data)
        for times in range(len(dataset) // batch_size):
            batch = iterator.next()
            assert len(batch) == batch_size
            # should be provided dataset + 1
            for index in range(len(batch)):
                assert batch[index] == dataset[times * batch_size + index] + 1

    def test_end_index(self):
        dataset = [data for data in range(50)]
        batch_size = 16
        iterator = decorable_multiprocess_iterator.DecorableMultiprocessIterator(dataset=dataset,
                                                                                 batch_size=batch_size,
                                                                                 repeat=False,
                                                                                 shuffle=False,
                                                                                 decor_fun=None,
                                                                                 end_index=20)
        batch = iterator.next()
        assert len(batch) == batch_size

        # Since end index is 20 second batch should contain only 20 - 16(batch_size) items
        batch = iterator.next()
        assert len(batch) == 4

    def test_epoch_detail(self):
        dataset = [data for data in range(50)]
        batch_size = 32
        iterator = decorable_multiprocess_iterator.DecorableMultiprocessIterator(dataset=dataset,
                                                                                 batch_size=batch_size,
                                                                                 repeat=True,
                                                                                 shuffle=False,
                                                                                 decor_fun=None,
                                                                                 end_index=40)
        iterator.next()
        assert iterator.epoch_detail <= 1.0

        iterator.next()
        iterator.next()
        assert 2.0 <= iterator.epoch_detail


if __name__ == '__main__':
    pytest.main()
