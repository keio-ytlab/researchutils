from researchutils import files

import chainer.serializers


def save_model(path, model):
    if files.file_exists(path):
        raise ValueError('File already exists in {}'.format(path))
    chainer.serializers.save_npz(path, model)


def load_model(path, model):
    if not files.file_exists(path):
        return model
    return chainer.serializers.load_npz(path, model)
