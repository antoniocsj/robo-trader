import os
import filecmp
from typing import Any

import numpy as np
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def write_json(_filename: str, _dict: dict):
    with open(_filename, 'w') as file:
        json.dump(_dict, file, indent=4)


def read_json(_filename: str) -> dict:
    if os.path.exists(_filename):
        with open(_filename, 'r') as file:
            _dict = json.load(file)
    else:
        print(f'ERRO. O arquivo {_filename} não foi encontrado.')
        exit(-1)
    return _dict


def update_settings(key: str, value: Any):
    settings = read_json('settings.json')
    settings[key] = value
    write_json('settings.json', settings)


def are_dir_trees_equal(dir1, dir2):
    """
    Compare two directories recursively. Files in each directory are
    assumed to be equal if their names and contents are equal.

    @param dir1: First directory path
    @param dir2: Second directory path

    @return: True if the directory trees are the same and
        there were no errors while accessing the directories or files,
        False otherwise.
   """

    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if len(dirs_cmp.left_only) > 0 or len(dirs_cmp.right_only) > 0 or len(dirs_cmp.funny_files) > 0:
        return False
    (_, mismatch, errors) = filecmp.cmpfiles(
        dir1, dir2, dirs_cmp.common_files, shallow=False)
    if len(mismatch) > 0 or len(errors) > 0:
        return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = os.path.join(dir1, common_dir)
        new_dir2 = os.path.join(dir2, common_dir)
        if not are_dir_trees_equal(new_dir1, new_dir2):
            return False
    return True


def get_list_sync_files(directory: str):
    _list = []
    all_files = os.listdir(directory)

    for filename in all_files:
        if filename.startswith('sync_cp_') and filename.endswith('.json'):
            _list.append(filename)

    return sorted(_list)


def load_sync_cp_file(_dirname: str, _filename: str) -> dict:
    _filepath = f'{_dirname}/{_filename}'
    if os.path.exists(_filepath):
        with open(_filepath, 'r') as file:
            cp = json.load(file)
    else:
        print(f'erro em load_sync_file(). arquivo {_filepath} não foi encontrado.')
        exit(-1)
    return cp


def write_train_config(train_config: dict):
    with open('train_config.json', 'w') as file:
        json.dump(train_config, file, indent=4, sort_keys=False, cls=NpEncoder)


def read_train_config() -> dict:
    _filename = 'train_config.json'
    if os.path.exists(_filename):
        with open(_filename, 'r') as file:
            _dict = json.load(file)
    else:
        print(f'ERRO. O arquivo {_filename} não foi encontrado.')
        exit(-1)
    return _dict


if __name__ == '__main__':
    pass
