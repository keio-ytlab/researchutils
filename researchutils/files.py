import argparse
import datetime
import json
import os
import sys


def file_exists(path):
    return os.path.exists(path)


def create_dir_if_not_exist(outdir):
    if file_exists(outdir):
        if not os.path.isdir(outdir):
            raise RuntimeError('{} is not a directory'.format(outdir))
        else:
            return
    os.makedirs(outdir)


def write_to_file(file_path, data):
    with open(file_path, 'w') as f:
        f.write(data)


def prepare_output_dir(base_dir, args, time_format='%Y-%m-%d-%H%M%S'):
    time_str = datetime.datetime.now().strftime(time_format)
    outdir = os.path.join(base_dir, time_str)
    create_dir_if_not_exist(outdir)

    # Save all the arguments
    args_file_path = os.path.join(outdir, 'args.txt')
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    write_to_file(args_file_path, args) 

    # Save the command
    argv_file_path = os.path.join(outdir, 'command.txt')
    argv = ' '.join(sys.argv)
    write_to_file(argv_file_path, argv) 

    return outdir
