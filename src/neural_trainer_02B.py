# faz um pesquisa mais profunda para encontrar o melhor random_seed.
# usa o arquivo train_log.json
# faz treinamentos com patience=15

import os
import copy
import time

from utils_filesystem import read_json, write_json
from neural_trainer_utils import train_model


params_nn = read_json('params_nn.json')
filename_basic = 'rs_basic_search.json'
filename_deeper = 'rs_deeper_search.json'


def load_rs_basic_search_json() -> dict:
    if os.path.exists(filename_basic):
        _dict = read_json(filename_basic)
        return _dict
    else:
        print(f'ERRO. o arquivo {filename_basic} não existe.')
        exit(-1)


def print_list(_list: list):
    for i in range(len(_list)):
        print(f'{i:03d} {_list[i]}')
    print('')


# def get_sorted_experiments():
#     train_log = load_train_log()
#     experiments = train_log['experiments']
#     sorted_exps_by_product_losses = sorted(experiments, key=lambda d: d['product'])
#
#     return sorted_exps_by_product_losses


def create_rs_deeper_search_json():
    """
    Cria o arquivo que guarda o status da pesquisa mais profunda do melhor random seed (rs_deeper_search.json).
    Antes de criar esse arquivo é necessário já existir um arquivo rs_basic_search.json válido contendo
    os experimentos dos treinos com diferentes random seeds com patience=patience_short.
    O arquivo rs_deeper_search.json terá muitos campos idênticos aos de rs_basic_search.json, porém ele será mais
    complexo.
    :return:
    """
    if not os.path.exists(filename_basic):
        print(f'ERRO. o arquivo {filename_basic} não existe.')
        exit(-1)
    else:
        rs_basic_search = load_rs_basic_search_json()

        timeframe_1, timeframe_2 = params_nn['timeframe'], rs_basic_search['timeframe']
        candle_input_type_1, candle_input_type_2 = params_nn['candle_input_type'], rs_basic_search['candle_input_type']
        n_steps_1, n_steps_2 = params_nn['n_steps'], rs_basic_search['n_steps']
        n_hidden_layers_1, n_hidden_layers_2 = params_nn['n_hidden_layers'], rs_basic_search['n_hidden_layers']

        # verificar se os parametros de rs_basic_search.json são idênticos aos definidos em params_nn.json
        if not (timeframe_1 == timeframe_2 and candle_input_type_1 == candle_input_type_2 and
                n_steps_1 == n_steps_2 and n_hidden_layers_1 == n_hidden_layers_2):
            print('ERRO. os parametros de rs_basic_search.json NÃO são idênticos aos definidos em params_nn.json')
            exit(-1)

        n_experiments: int = rs_basic_search['n_experiments']
        random_seed_max = params_nn['random_seed_max']

        if n_experiments < random_seed_max:
            print(f'ERRO! o arquivo {filename_basic} indica que o scan dos random seeds não foi completado ainda.')
            print(f'n_experiments ({n_experiments}) < random_seed_max ({random_seed_max})')

            # se houver um arquivo rs_deeper_search.json, delete-o, pois ele não está valido, um vez que
            # rs_basic_search.json indica que a busca básica ainda está incompleta.
            if os.path.exists(filename_deeper):
                print(f'removendo arquivo inválido: {filename_deeper}')
                os.remove(filename_deeper)

            exit(-1)

    if not os.path.exists(filename_deeper):
        print(f'o arquivo {filename_deeper} não existe ainda. será criado agora.')

        # o arquivo rs_deeper_search.json será inicialmente quase idêntico ao arquivo rs_basic_search.json, a
        # diferença está no campo 'experiments' que será renomeado para 'sorted_experiments' e conterá a lista dos
        # experimentos ordenada pelo 'campo product'.
        _dict = copy.deepcopy(rs_basic_search)
        experiments = _dict['experiments']
        sorted_experiments = sorted(experiments, key=lambda d: d['product'])
        del _dict['experiments']
        _dict['sorted_experiments'] = sorted_experiments
        write_json(filename_deeper, _dict)
    else:
        print(f'o arquivo {filename_deeper} já existe. continuando a pesquisa do melhor random seed.')


def load_rs_deeper_search_json() -> dict:
    if os.path.exists(filename_deeper):
        _dict = read_json(filename_deeper)
        return _dict
    else:
        print(f'ERRO. o arquivo {filename_deeper} não existe.')
        exit(-1)


def update_rs_deeper_search_json(rs_deeper_search: dict):
    if os.path.exists(filename_deeper):
        write_json(filename_deeper, rs_deeper_search)
    else:
        print(f'ERRO. o arquivo {filename_deeper} não existe.')
        exit(-1)


def nn_train_search_best_random_seed():
    create_rs_deeper_search_json()


if __name__ == '__main__':
    nn_train_search_best_random_seed()
