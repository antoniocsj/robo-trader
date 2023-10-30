# faz uma pesquisa básica por random seeds, isto é, executa vários treinamentos com random_seeds diferentes e com
# patience=params_rs_search['patience_short'];
# guarda os principais resultados de cada pesquisa em rs_basic_search.json;

import os
import time

from utils_checks import initial_compliance_checks
from utils_filesystem import read_json, write_json
from neural_trainer_utils import train_model, get_time_break_from_timeframe


# tanto a pesquisa básica quanto a pesquisa profunda, usam os parâmetros definidos em params_rs_search.json
params_rs_search = read_json('params_rs_search.json')
filename_basic = 'rs_basic_search.json'


def create_rs_basic_search_json():
    if not os.path.exists(filename_basic):
        print(f'o arquivo {filename_basic} não existe ainda. será criado agora.')
        _dict = {
            'n_basic_experiments': 0,
            'symbol_out': '',
            'timeframe': '',
            'candle_input_type': '',
            'n_steps': 0,
            'n_hidden_layers': 0,
            'candle_output_type': '',
            'n_symbols': 0,
            'symbols': [],
            'n_samples_train': 0,
            'validation_split': 0.0,
            'samples_test_ratio': 0.0,
            'n_samples_test': 0,
            'max_n_epochs': 0,
            'patience': 0,
            'datetime_start': '',
            'datetime_end': '',
            'basic_experiments': []
        }
        write_json(filename_basic, _dict)
    else:
        print(f'o arquivo {filename_basic} já existe. continuando os experimentos.')


def load_rs_basic_search_json() -> dict:
    if os.path.exists(filename_basic):
        _dict = read_json(filename_basic)
        return _dict
    else:
        print(f'ERRO. o arquivo {filename_basic} não existe.')
        exit(-1)


def update_rs_basic_search_json(_dict: dict):
    if os.path.exists(filename_basic):
        write_json(filename_basic, _dict)
    else:
        print(f'ERRO. o arquivo {filename_basic} não existe.')
        exit(-1)


def nn_train_rs_basic_search():
    print('nn_train_rs_basic_search')

    initial_compliance_checks()

    settings = read_json('settings.json')
    create_rs_basic_search_json()
    time_break_secs: int = get_time_break_from_timeframe(settings['timeframe'])
    random_seed_max: int = params_rs_search['random_seed_max']

    while True:
        rs_basic_search = load_rs_basic_search_json()
        n_basic_experiments: int = rs_basic_search['n_basic_experiments']
        index = n_basic_experiments + 1

        if index > random_seed_max:
            print(f'nn_train_rs_basic_search: CONCLUÍDO. n_basic_experiments = {n_basic_experiments} == '
                  f'params_rs_search["random_seed_max"] ({random_seed_max})')
            break

        train_config = train_model(settings, params_rs_search=params_rs_search, seed=index, patience_style='short')

        whole_set_train_loss_eval = train_config['whole_set_train_loss_eval']
        test_loss_eval = train_config['test_loss_eval']
        losses_product = whole_set_train_loss_eval * test_loss_eval

        rs_basic_search['n_basic_experiments'] = index

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

        rs_basic_search.update(_dict)

        log = {
            'random_seed': index,
            'effective_n_epochs': train_config['effective_n_epochs'],
            'whole_set_train_loss': whole_set_train_loss_eval,
            'test_loss': test_loss_eval,
            'losses_product': losses_product
        }
        rs_basic_search['basic_experiments'].append(log)

        rs_basic_search['n_basic_experiments'] = len(rs_basic_search['basic_experiments'])
        update_rs_basic_search_json(rs_basic_search)

        # sempre espera alguns segundos para não superaquecer a placa de vídeo
        # o 'if' é para não precisar esperar após o último experimento
        if index < random_seed_max:
            print(f'esperando por {time_break_secs} segundos')
            time.sleep(time_break_secs)
        else:
            break


if __name__ == '__main__':
    _settings = read_json('settings.json')
    _settings['timeframe'] = params_rs_search['timeframe']
    _settings['candle_input_type'] = params_rs_search['candle_input_type']
    _settings['random_seed'] = 1
    write_json('settings.json', _settings)

    nn_train_rs_basic_search()
