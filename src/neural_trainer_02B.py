# faz um pesquisa mais profunda para encontrar o melhor random_seed.
# usa o arquivo train_log.json
# faz treinamentos com patience=15

import os
import time

from utils_filesystem import read_json, write_json
from neural_trainer_utils import train_model


params_nn = read_json('params_nn.json')


def load_train_log() -> dict:
    import os
    from utils_filesystem import read_json
    _filename = 'train_log.json'
    if os.path.exists(_filename):
        _dict = read_json(_filename)
        return _dict
    else:
        print(f'ERRO. o arquivo {_filename} não existe.')
        exit(-1)


def print_list(_list: list):
    for i in range(len(_list)):
        print(f'{i:03d} {_list[i]}')
    print('')


def get_sorted_experiments():
    train_log = load_train_log()
    experiments = train_log['experiments']
    sorted_exps_by_product_losses = sorted(experiments, key=lambda d: d['product'])

    return sorted_exps_by_product_losses


def create_rs_search_status():
    """
    Cria o arquivo que guarda o status da pesquisa do melhor random seed (rs_search_status.json).
    Antes de criar esse arquivo é necessário já existir um arquivo train_log.json válido contendo
    os experimentos dos treinos com diferentes random seeds.
    :return:
    """
    import os

    _filename_1 = 'train_log.json'
    if not os.path.exists(_filename_1):
        print(f'ERRO. o arquivo {_filename_1} não existe.')
        exit(-1)
    else:
        train_log = load_train_log()
        n_experiments: int = train_log['n_experiments']
        rs_start = params_nn['random_seed_start']
        rs_end = params_nn['random_seed_end']
        if n_experiments < rs_end - rs_start + 1:
            print(f'ERRO! o arquivo {_filename_1} indica que o scan dos random seeds não foi completado ainda.')
            exit(-1)

    _filename = 'rs_search_status.json'
    if not os.path.exists(_filename):
        print(f'o arquivo {_filename} não existe ainda. será criado agora.')
        _dict = {}
        write_json(_filename, _dict)
    else:
        print(f'o arquivo {_filename} já existe. continuando a pesquisa do melhor random seed.')


def load_rs_search_status() -> dict:
    _filename = 'rs_search_status.json'
    if os.path.exists(_filename):
        _dict = read_json(_filename)
        return _dict
    else:
        print(f'ERRO. o arquivo {_filename} não existe.')
        exit(-1)


def update_rs_search_status(rs_search_status: dict):
    _filename = 'rs_search_status.json'
    if os.path.exists(_filename):
        write_json(_filename, rs_search_status)
    else:
        print(f'ERRO. o arquivo {_filename} não existe.')
        exit(-1)


def nn_train_search_best_random_seed():
    create_rs_search_status()


if __name__ == '__main__':
    nn_train_search_best_random_seed()
