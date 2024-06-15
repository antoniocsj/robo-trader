import os
import filecmp
import shutil
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


def write_json(filename: str, _dict: dict):
    with open(filename, 'w') as file:
        json.dump(_dict, file, indent=4)


def read_json(filename: str) -> dict:
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            _dict = json.load(file)
    else:
        print(f'ERRO. O arquivo {filename} não foi encontrado.')
        exit(-1)
    return _dict


def update_settings(key: str, value: Any):
    working_dir = os.getcwd()
    settings_filepath = os.path.join(working_dir, 'settings.json')
    settings = read_json(settings_filepath)
    settings[key] = value
    write_json(settings_filepath, settings)


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


def are_files_equal(filenames: list[str], src_dir: str, dst_dir: str):
    for filename in filenames:
        filepath_src = f'{src_dir}/{filename}'
        filepath_dst = f'{dst_dir}/{filename}'

        result = filecmp.cmp(filepath_src, filepath_dst, shallow=False)

        if not result:
            return False

    return True


def get_list_sync_files(directory: str):
    _list = []
    all_files = os.listdir(directory)

    for filename in all_files:
        if filename.startswith('sync_cp') and filename.endswith('.json'):
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
    train_config_filename = 'train_config.json'

    with open(train_config_filename, 'w') as file:
        json.dump(train_config, file, indent=4, sort_keys=False, cls=NpEncoder)

    print(f'O arquivo {train_config_filename} foi gravado com SUCESSO.')


def write_train_config2(directory: str, train_config: dict):
    train_config_filename = 'train_config.json'
    filepath = os.path.join(directory, train_config_filename)

    with open(filepath, 'w') as file:
        json.dump(train_config, file, indent=4, sort_keys=False, cls=NpEncoder)

    print(f'O arquivo {filepath} foi gravado com SUCESSO.')


def read_train_config() -> dict:
    train_config_filename = 'train_config.json'

    if os.path.exists(train_config_filename):
        with open(train_config_filename, 'r') as file:
            _dict = json.load(file)
    else:
        print(f'ERRO. O arquivo {train_config_filename} não foi encontrado.')
        exit(-1)

    return _dict


def read_train_config2(directory: str) -> dict:
    train_config_filename = 'train_config.json'
    filepath = os.path.join(directory, train_config_filename)

    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            _dict = json.load(file)
    else:
        print(f'ERRO. O arquivo {filepath} não foi encontrado.')
        exit(-1)

    return _dict


def make_synch_backup(src_dir: str, dst_dir: str):
    print(f'copiando os arquivos sincronizados para o diretório {dst_dir}')
    if os.path.exists(dst_dir):
        print(f'o diretório {dst_dir} já existe. será substituído.')
        shutil.rmtree(dst_dir)

    shutil.copytree(src_dir, dst_dir)
    if are_dir_trees_equal(src_dir, dst_dir):
        print('Backup efetuado e verificado com SUCESSO!')
        # aproveita e copia o arquivo final de checkpoint de sincronização 'sync_cp.json' para dst_dir também
        _list = get_list_sync_files('.')

        if len(_list) > 0:
            _sync_filename = _list[0]
            _sync_filepath_copy = f'{dst_dir}/{_sync_filename}'
            shutil.copy(_sync_filename, _sync_filepath_copy)
    else:
        print('ERRO ao fazer o backup.')
        exit(-1)


def copy_files(filenames: list[str], src_dir: str, dst_dir: str, clear_dst=False):
    print(f'copiando de {src_dir} para {dst_dir} os seguintes arquivos:')
    print(f'{filenames}')

    if clear_dst and os.path.exists(dst_dir):
        print(f'o diretório {dst_dir} já existe. será substituído.')
        shutil.rmtree(dst_dir)

    for filename in filenames:
        filepath_src = f'{src_dir}/{filename}'
        filepath_dst = f'{dst_dir}/{filename}'
        shutil.copy(filepath_src, filepath_dst)

    if are_files_equal(filenames, src_dir, dst_dir):
        print('A cópia dos arquivos foi efetuada e verificada com SUCESSO!')
    else:
        print('ERRO ao fazer a cópia dos arquivos.')
        exit(-1)


def remove_files(filenames: list[str], directory: str):
    print(f'removendo de {directory} os seguintes arquivos:')
    print(f'{filenames}')

    if not os.path.exists(directory):
        print(f'ERRO. O diretório {directory} não existe. abortando.')
        exit(-1)

    for filename in filenames:
        filepath = f'{directory}/{filename}'
        os.remove(filepath)

    print(f'arquivos removidos com sucesso.')


def reset_dir(dirname: str):
    print(f'o diretório {dirname} foi resetado.')
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


def copy_dir(src_dir: str, dst_dir: str):
    """
    Realiza uma cópia do diretório src_dir dentro do diretório dst_dir.
    Parameters
    ----------
    src_dir
    dst_dir

    Returns
    -------

    """
    filenames = os.listdir(src_dir)

    print(f'copiando o diretório {src_dir} para {dst_dir}:')

    # verifique se os diretórios fonte e destino existem
    if not os.path.exists(src_dir):
        print('ERRO')
        exit(-1)
    if not os.path.exists(dst_dir):
        print('ERRO')
        exit(-1)

    # verifique se src_dir já existe dentro de dst_dir, caso exista resete-o
    inner_dir = os.path.join(dst_dir, os.path.basename(src_dir))
    reset_dir(inner_dir)

    for filename in filenames:
        filepath_src = f'{src_dir}/{filename}'
        filepath_dst = f'{inner_dir}/{filename}'
        shutil.copy(filepath_src, filepath_dst)

    if are_files_equal(filenames, src_dir, inner_dir):
        print('A cópia dos arquivos foi efetuada e verificada com SUCESSO!')
    else:
        print('ERRO ao fazer a cópia dos arquivos.')
        exit(-1)


if __name__ == '__main__':
    pass
