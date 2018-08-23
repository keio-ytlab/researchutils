from researchutils import files
from mock import patch
import argparse
import pytest


class TestFiles(object):
    def test_file_exists(self):
        with patch('os.path.exists', return_value=True) as mock_exists:
            test_file = "test"
            assert files.file_exists(test_file) == True

    def test_file_does_not_exist(self):
        with patch('os.path.exists', return_value=False) as mock_exists:
            test_file = "test"
            assert files.file_exists(test_file) == False

    def test_create_dir_if_not_exist(self):
        with patch('os.path.exists', return_value=False) as mock_exists, patch('os.makedirs') as mock_mkdirs:
            test_file = "test"
            files.create_dir_if_not_exist(test_file)

            mock_exists.assert_called_once()
            mock_mkdirs.assert_called_once()

    def test_create_dir_when_exists(self):
        with patch('os.path.exists', return_value=True) as mock_exists, patch('os.makedirs') as mock_mkdirs:
            with patch('os.path.isdir', return_value=True):
                test_file = "test"
                files.create_dir_if_not_exist(test_file)

                mock_exists.assert_called_once()
                mock_mkdirs.assert_not_called()

    def test_create_dir_when_target_is_not_directory(self):
        with patch('os.path.exists', return_value=True) as mock_exists, patch('os.makedirs') as mock_mkdirs:
            with patch('os.path.isdir', return_value=False):
                with pytest.raises(RuntimeError):
                    test_file = "test"
                    files.create_dir_if_not_exist(test_file)

                    mock_exists.assert_called_once()
                    mock_mkdirs.assert_not_called()

    def test_prepare_output_dir(self):
        kwargs = dict(test1='test1', test2='test2')
        args = argparse.Namespace(**kwargs)
        with patch('os.path.exists', return_value=False), patch('researchutils.files.write_to_file'):
            base_dir = 'base/'
            output_dir = files.prepare_output_dir(base_dir, args)
            assert not len(output_dir) == 0


if __name__ == '__main__':
    pytest.main()
