# faz um pesquisa mais profunda para encontrar o melhor random_seed.
# usa o arquivo train_log.json
# faz treinamentos com patience=15

import os
import copy
import time

from src.utils.utils_checks import initial_compliance_checks
from src.utils.utils_filesystem import read_json, write_json
from neural_trainer_utils import train_model, get_time_break_from_timeframe
from src.setups import run_setup
from src.neural_trainer_02A import load_rs_basic_search_json

params_rs_search = read_json('params_rs_search.json')
filename_basic = 'rs_basic_search.json'
filename_deep = 'rs_deep_search.json'

# indica o número de experimentos que serão selecionados do início da lista ordenada dos experimentos
# básicos (patience_short) para uma busca mais profunda (patience_long) do random_seed que gera o menor
# 'product_loss' no treinamento e teste da rede neural.
deep_search_range = params_rs_search['deep_search_range']


def print_list(_list: list):
    for i in range(len(_list)):
        print(f'{i:03d} {_list[i]}')
    print('')


def create_rs_deep_search_json(results_dir: str):
    """
    Cria o arquivo que guarda o status da pesquisa mais profunda do melhor random seed (rs_deeper_search.json).
    Antes de criar esse arquivo é necessário já existir um arquivo rs_basic_search.json válido contendo
    os experimentos dos treinos com diferentes random seeds com patience=patience_short.
    O arquivo rs_deeper_search.json terá muitos campos idênticos aos de rs_basic_search.json, porém ele será mais
    complexo.
    :return:
    """
    filepath_basic = os.path.join(results_dir, filename_basic)
    filepath_deep = os.path.join(results_dir, filename_deep)

    # é necessário que o arquivo rs_basic_search.json exista, pois isso indica que a busca básica já foi completada.
    if not os.path.exists(filepath_basic):
        print(f'ERRO. o arquivo {filepath_basic} não existe.')
        exit(-1)

    rs_basic_search = load_rs_basic_search_json(results_dir)

    timeframe_1, timeframe_2 = params_rs_search['timeframe'], rs_basic_search['timeframe']
    candle_input_type_1, candle_input_type_2 = params_rs_search['candle_input_type'], rs_basic_search['candle_input_type']

    n_steps_1, n_steps_2 = params_rs_search['n_steps'], rs_basic_search['n_steps']
    n_hidden_layers_1, n_hidden_layers_2 = params_rs_search['n_hidden_layers'], rs_basic_search['n_hidden_layers']

    # verificar se os parametros de rs_basic_search.json são idênticos aos definidos em params_rs_search.json
    if not (timeframe_1 == timeframe_2 and candle_input_type_1 == candle_input_type_2 and
            n_steps_1 == n_steps_2 and n_hidden_layers_1 == n_hidden_layers_2):
        print('ERRO. os parametros de rs_basic_search.json NÃO são idênticos aos definidos em params_rs_search.json')
        exit(-1)

    n_basic_experiments: int = rs_basic_search['n_basic_experiments']
    random_seed_max = params_rs_search['random_seed_max']

    # aborta se a busca básica ainda não foi completada
    if n_basic_experiments < random_seed_max:
        print(f'ERRO! o arquivo {filename_basic} indica que o scan dos random seeds não foi completado ainda.')
        print(f'n_basic_experiments ({n_basic_experiments}) < random_seed_max ({random_seed_max})')
        # se houver um arquivo rs_deeper_search.json, delete-o, pois ele não está valido, um vez que
        # rs_basic_search.json indica que a busca básica ainda está incompleta.
        if os.path.exists(filepath_deep):
            print(f'há um arquivo remanescente de busca aprofundada: {filepath_deep}. deletando-o.')
            os.remove(filepath_deep)
        exit(-1)

    # se o arquivo rs_deeper_search.json não existe, então crie-o.
    if not os.path.exists(filepath_deep):
        print(f'o arquivo {filepath_deep} não existe ainda. será criado agora.')

        # o arquivo rs_deeper_search.json será inicialmente quase idêntico ao arquivo rs_basic_search.json, as
        # diferenças serão as seguintes
        # - o campo 'experiments' será renomeado para 'sorted_basic_experiments', e conterá a mesma lista de
        # experimentos, porém estará ordenada pelo campo 'product' (product losses);
        # o campo 'patience' terá seu valor ajustado para o valor de params_rs_search['patience_long'];
        # novo campo 'sorted_experiments_deeper', que guardará a lista ordenada dos experimentos mais profundos
        # (patience_long). essa lista começa vazia inicialmente, e crescerá até atingir o número de elementos
        # definido por 'deep_search_range';
        # novo campo 'deep_search_range', que indica tamanho máximo da lista 'sorted_deeper_experiments';
        # novo campo n_deeper_experiments, que indica o tamanho atual da lista 'sorted_deeper_experiments';
        _dict = copy.deepcopy(rs_basic_search)
        basic_experiments = _dict['basic_experiments']
        sorted_basic_experiments = sorted(basic_experiments, key=lambda d: d['losses_product'])
        _dict['patience'] = params_rs_search['patience_long']
        _dict['sorted_basic_experiments'] = sorted_basic_experiments
        _dict['deep_search_range'] = deep_search_range
        _dict['n_deep_experiments'] = 0
        _dict['deep_experiments'] = []
        _dict['sorted_deep_experiments'] = []
        _dict['best_deep_random_seed'] = -1
        write_json(filepath_deep, _dict)
    else:
        print(f'o arquivo {filepath_deep} já existe. continuando a pesquisa do melhor random seed.')


def load_rs_deep_search_json(results_dir: str) -> dict:
    filepath_deep = os.path.join(results_dir, filename_deep)
    if os.path.exists(filepath_deep):
        _dict = read_json(filepath_deep)
        return _dict
    else:
        print(f'ERRO. o arquivo {filepath_deep} não existe.')
        exit(-1)


def update_rs_deep_search_json(results_dir: str, _dict: dict):
    filepath_deep = os.path.join(results_dir, filename_deep)
    if os.path.exists(filepath_deep):
        write_json(filepath_deep, _dict)
    else:
        print(f'ERRO. o arquivo {filepath_deep} não existe.')
        exit(-1)


def nn_train_rs_deep_search():
    print('nn_train_rs_deep_search')

    working_dir = os.getcwd()
    initial_compliance_checks(working_dir)

    settings = read_json('settings.json')
    time_break_secs: int = get_time_break_from_timeframe(settings['timeframe']) * 2
    # time_break_secs = 0

    results_dir = os.path.join(working_dir, settings['results_dir'])

    # se o diretório 'results' não já existe, então aborte, pois a busca básica ainda não foi iniciada.
    if not os.path.exists(results_dir):
        print('ERRO. o diretório de resultados não existe. a busca básica ainda não foi iniciada.')
        exit(-1)

    create_rs_deep_search_json(results_dir)
    rs_deep_search = load_rs_deep_search_json(results_dir)

    # refaz o treinamento dos N primeiros experimentos de 'sorted_basic_experiments', onde N = deep_search_range,
    # porém, desta vez, o treinamento das redes neurais será com 'patience' = patience_long
    N = rs_deep_search['deep_search_range']
    sorted_basic_experiments = rs_deep_search['sorted_basic_experiments']

    # verifica se deep_search_range é maior que o número de experimentos básicos
    if N > len(sorted_basic_experiments):
        print(f'ERRO. deep_search_range ({N}) > len(sorted_basic_experiments) ({len(sorted_basic_experiments)})')
        exit(-1)

    first_basic_experiments = sorted_basic_experiments[0:deep_search_range]
    deep_experiments = rs_deep_search['deep_experiments']
    len_deep_experiments = len(deep_experiments)

    if len_deep_experiments >= N:
        print(f'a busca profunda já terminou. deep_search_range = {N}')
        sorted_deep_experiments = rs_deep_search['sorted_deep_experiments']
        best_deep_random_seed = sorted_deep_experiments[0]['random_seed']
        rs_deep_search['best_deep_random_seed'] = best_deep_random_seed
        update_rs_deep_search_json(results_dir, rs_deep_search)
        print(f'best_deep_random_seed = {best_deep_random_seed}')
        print(f'{sorted_deep_experiments[0]}')
        exit(0)

    for i in range(len_deep_experiments, N):
        e = first_basic_experiments[i]
        print(e)
        seed: int = e['random_seed']

        train_config = train_model(working_dir, settings, params_rs_search, seed, patience_style='long')

        whole_set_train_loss_eval = train_config['whole_set_train_loss_eval']
        test_loss_eval = train_config['test_loss_eval']
        losses_product = whole_set_train_loss_eval * test_loss_eval

        _dict = {
            'symbol_out': train_config['symbol_out'],
            'timeframe': train_config['timeframe'],
            'candle_input_type': train_config['candle_input_type'],
            'n_steps': train_config['n_steps'],
            'n_hidden_layers': train_config['n_hidden_layers'],
            'candle_output_type': train_config['candle_output_type'],
            'n_symbols': train_config['n_symbols'],
            'symbols': train_config['symbols'],
            'n_samples_train': train_config['n_samples_train'],
            'validation_split': train_config['validation_split'],
            'samples_test_ratio': train_config['samples_test_ratio'],
            'n_samples_test': train_config['n_samples_test'],
            'max_n_epochs': train_config['max_n_epochs'],
            'patience': train_config['patience'],
            'datetime_start': train_config['datetime_start'],
            'datetime_end': train_config['datetime_end']
        }

        rs_deep_search.update(_dict)

        log = {
            'random_seed': seed,
            'effective_n_epochs': train_config['effective_n_epochs'],
            'whole_set_train_loss': whole_set_train_loss_eval,
            'test_loss': test_loss_eval,
            'losses_product': losses_product
        }
        rs_deep_search['deep_experiments'].append(log)

        rs_deep_search['n_deep_experiments'] = len(rs_deep_search['deep_experiments'])
        rs_deep_search['sorted_deep_experiments'] = sorted(rs_deep_search['deep_experiments'],
                                                           key=lambda d: d['losses_product'])
        update_rs_deep_search_json(results_dir, rs_deep_search)

        # sempre espera alguns segundos para não superaquecer a placa de vídeo
        # o 'if' é para não precisar esperar após o último experimento
        if i < N - 1:
            print(f'esperando por {time_break_secs} segundos')
            time.sleep(time_break_secs)
        else:
            break

    rs_deep_search = load_rs_deep_search_json(results_dir)
    sorted_deep_experiments = rs_deep_search['sorted_deep_experiments']
    best_deep_random_seed = sorted_deep_experiments[0]['random_seed']
    rs_deep_search['best_deep_random_seed'] = best_deep_random_seed
    update_rs_deep_search_json(results_dir, rs_deep_search)
    print(f'best_deep_random_seed = {best_deep_random_seed}')
    print(f'{sorted_deep_experiments[0]}')
    pass


if __name__ == '__main__':
    run_setup()
    nn_train_rs_deep_search()
