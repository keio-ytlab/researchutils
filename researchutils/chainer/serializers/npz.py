from researchutils import files

import chainer.serializers


def save_model(path, model):
    """
    Save model as an npz file to given path

    Parameters
    -------
    path : string
        path of the model to be saved
    model : chainer.Link
        model to save parameters

    Raises
    -------
    ValueError
        File already exists
    """
    if files.file_exists(path):
        raise ValueError('File already exists in {}'.format(path))
    chainer.serializers.save_npz(path, model)


def load_model(path, model):
    """
    Load model from the npz file of given path

    Parameters
    ------
    path : string
        path of the saved model

    Returns
    ------
    model : chainer.Link
        model with parameters initialized from loaded file
        if the file does not exist, then will return given model without any changes
    """
    if not files.file_exists(path):
        return model
    return chainer.serializers.load_npz(path, model)
