from researchutils.chainer.serializers import npz
from researchutils import files
from chainer.links.model.vision import vgg
from chainer import Chain
from chainer.training import Trainer

import os

import tempfile
import pytest
from mock import patch
from mock import Mock


class MockModel(Chain):
    def __init__(self):
        super(MockModel, self).__init__()


class TestNpz(object):
    def test_save_model_no_file(self):
        with patch('chainer.serializers.save_npz') as mock_save_npz:
            target_path = os.path.join(tempfile.tempdir, 'test.npz')
            if files.file_exists(target_path):
                os.remove(target_path)
            test_model = MockModel()
            npz.save_model(target_path, test_model)

            assert mock_save_npz.call_count == 1

    def test_save_model_file_exists(self):
        with patch('chainer.serializers.save_npz') as _:
            target_path = os.path.join(tempfile.tempdir, 'test.npz')
            with open(target_path, "a+") as f:
                test_model = MockModel()
                with pytest.raises(ValueError):
                    npz.save_model(target_path, test_model)
                    os.remove(target_path)

    def test_load_model_no_file(self):
        with patch('chainer.serializers.load_npz') as _:
            target_path = os.path.join(tempfile.tempdir, 'test.npz')
            if files.file_exists(target_path):
                os.remove(target_path)
            test_model = MockModel()
            loaded_model = npz.load_model(target_path, test_model)

            assert test_model == loaded_model

    def test_load_model_file_exists(self):
        with patch('chainer.serializers.load_npz') as mock_load_npz:
            target_path = os.path.join(tempfile.tempdir, 'test.npz')
            with open(target_path, "a+") as f:
                test_model = MockModel()
                npz.load_model(target_path, test_model)
                os.remove(target_path)

            assert mock_load_npz.call_count == 1

    def test_load_snapshot_no_file(self):
        with patch('chainer.serializers.load_npz') as _:
            target_path = os.path.join(tempfile.tempdir, 'test.npz')
            if files.file_exists(target_path):
                os.remove(target_path)
            test_trainer = Mock(spec=Trainer)
            loaded_trainer = npz.load_snapshot(target_path, test_trainer)

            assert test_trainer == loaded_trainer

    def test_load_snapshot_file_exists(self):
        with patch('chainer.serializers.load_npz') as mock_load_npz:
            target_path = os.path.join(tempfile.tempdir, 'test.npz')
            with open(target_path, "a+") as f:
                test_trainer = Mock(spec=Trainer)
                npz.load_snapshot(target_path, test_trainer)
                os.remove(target_path)

            assert mock_load_npz.call_count == 1

if __name__ == "__main__":
    pytest.main()
