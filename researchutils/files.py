import argparse
import datetime
import json
import os
import sys
import six.moves.cPickle as pickle


def file_exists(path):
    """
    Check file existence on given path

    Parameters
    -------
    path : string
        Path of the file to check existence

    Returns
    -------
    file_existence : bool
        True if file exists otherwise False        
    """
    return os.path.exists(path)


def create_dir_if_not_exist(outdir):
    """
    Check directory existence and creates new directory if not exist

    Parameters
    -------
    outdir : string
        Path of the file to create directory

    Raises
    ------
    RuntimeError
        File exists in outdir but it is not a directory
    """
    if file_exists(outdir):
        if not os.path.isdir(outdir):
            raise RuntimeError('{} is not a directory'.format(outdir))
        else:
            return
    os.makedirs(outdir)


def write_text_to_file(file_path, data):
    """
    Write given text data to file

    Parameters
    -------
    file_path : string
        Path of the file to write data
    data: string
        Text to write to the file
    """
    with open(file_path, 'w') as f:
        f.write(data)


def read_text_from_file(file_path):
    """
    Read given file as text

    Parameters
    -------
    file_path : string
        Path of the file to read data

    Returns
    -------
    data: string
        Text read from the file
    """
    with open(file_path, 'r') as f:
        return f.read()


def save_pickle(file_path, data):
    """
    Pickle given data to file

    Parameters
    -------
    file_path : string
        Path of the file to pickle data
    data : data to pickle
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    """
    Load pickled data from file

    Parameters
    -------
    file_path : string
        Path of the file to load pickled data
    
    Returns
    -------
    data : data pickled in file
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f, encoding='bytes')


def prepare_output_dir(base_dir, args, time_format='%Y-%m-%d-%H%M%S'):
    """
    Prepare a directory with current datetime as name.
    Created directory contains the command and args when the script was called as text file.

    Parameters
    -------
    base_dir : string
        Path of the directory to save data
    args : dictionary
        Arguments when the python script was called
    time_format : string
        Datetime format string for naming directory to save data
    
    Returns
    -------
    out_dir : directory to save data
    """ 
    time_str = datetime.datetime.now().strftime(time_format)
    outdir = os.path.join(base_dir, time_str)
    create_dir_if_not_exist(outdir)

    # Save all the arguments
    args_file_path = os.path.join(outdir, 'args.txt')
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    write_text_to_file(args_file_path, json.dumps(args))

    # Save the command
    argv_file_path = os.path.join(outdir, 'command.txt')
    argv = ' '.join(sys.argv)
    write_text_to_file(argv_file_path, argv)

    return outdir
