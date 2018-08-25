import argparse
import datetime
import json
import os
import sys
import six.moves.cPickle as pickle


def file_exists(path):
    return os.path.exists(path)


def create_dir_if_not_exist(outdir):
    if file_exists(outdir):
        if not os.path.isdir(outdir):
            raise RuntimeError('{} is not a directory'.format(outdir))
        else:
            return
    os.makedirs(outdir)


def write_text_to_file(file_path, data):
    with open(file_path, 'w') as f:
        f.write(data)


def read_text_from_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def save_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def prepare_output_dir(base_dir, args, time_format='%Y-%m-%d-%H%M%S'):
    time_str = datetime.datetime.now().strftime(time_format)
    outdir = os.path.join(base_dir, time_str)
    create_dir_if_not_exist(outdir)

    # Save all the arguments
    args_file_path = os.path.join(outdir, 'args.txt')
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    write_text_to_file(args_file_path, args)

    # Save the command
    argv_file_path = os.path.join(outdir, 'command.txt')
    argv = ' '.join(sys.argv)
    write_text_to_file(argv_file_path, argv)

    return outdir
