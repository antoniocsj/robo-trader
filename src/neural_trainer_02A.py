# faz uma pesquisa básica por random seeds, isto é, executa vários treinamentos com random_seeds diferentes e com
# patience=params_rs_search['patience_short'];
# guarda os principais resultados de cada pesquisa em rs_basic_search.json;

import os
import time
import shutil

from src.utils.utils_checks import initial_compliance_checks
from src.utils.utils_filesystem import read_json, write_json
from neural_trainer_utils import train_model, get_time_break_from_timeframe
from src.setups import run_setup


# tanto a pesquisa básica quanto a pesquisa profunda, usam os parâmetros definidos em params_rs_search.json
params_rs_search = read_json('params_rs_search.json')
filename_basic = 'rs_basic_search.json'


def create_rs_basic_search_json(results_dir: str):
    filepath_basic = os.path.join(results_dir, filename_basic)

    if not os.path.exists(filepath_basic):
        print(f'o arquivo {filepath_basic} não existe ainda. será criado agora.')
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
            'basic_experiments': [],
            'sorted_basic_experiments': []
        }
        write_json(filepath_basic, _dict)
    else:
        print(f'o arquivo {filepath_basic} já existe. continuando os experimentos.')


def load_rs_basic_search_json(results_dir: str) -> dict:
    filepath_basic = os.path.join(results_dir, filename_basic)
    if os.path.exists(filepath_basic):
        _dict = read_json(filepath_basic)
        return _dict
    else:
        print(f'ERRO. o arquivo {filepath_basic} não foi encontrado no diretório {results_dir}')
        exit(-1)


def try_load_rs_basic_search_json(results_dir: str) -> dict or None:
    filepath_basic = os.path.join(results_dir, filename_basic)
    if os.path.exists(filepath_basic):
        _dict = read_json(filepath_basic)
        return _dict
    else:
        print(f'AVISO. o arquivo {filepath_basic} não foi encontrado no diretório {results_dir}')
        return None


def update_rs_basic_search_json(results_dir: str, _dict: dict):
    filepath_basic = os.path.join(results_dir, filename_basic)
    if os.path.exists(filepath_basic):
        write_json(filepath_basic, _dict)
    else:
        print(f'ERRO. o arquivo {filepath_basic} não existe.')
        exit(-1)


def nn_train_rs_basic_search():
    print('nn_train_rs_basic_search')

    working_dir = os.getcwd()
    initial_compliance_checks(working_dir)

    settings = read_json('settings.json')
    time_break_secs: int = get_time_break_from_timeframe(settings['timeframe'])
    # time_break_secs = 0
    random_seed_max: int = params_rs_search['random_seed_max']

    # Create save directory if not exists
    results_dir = os.path.join(working_dir, settings['results_dir'])

    if os.path.exists(results_dir):
        # se o diretório 'results' já existe, verifica se a busca básica já foi concluída.
        rs_basic_search = try_load_rs_basic_search_json(results_dir)
        # se a busca básica ainda não foi concluída, então continua os experimentos.
        if rs_basic_search and rs_basic_search['n_basic_experiments'] < random_seed_max:
            print('o arquivo rs_basic_search.json já existe no diretório de resultados. continuando os experimentos.')
        # se a busca básica já foi concluída, então apaga o diretório de resultados e reinicia a busca.
        elif rs_basic_search and rs_basic_search['n_basic_experiments'] >= random_seed_max:
            print(f'o arquivo rs_basic_search.json indica que a busca básica já foi concluída. '
                  f'n_basic_experiments = {rs_basic_search["n_basic_experiments"]} == '
                  f'params_rs_search["random_seed_max"] ({random_seed_max})')
            shutil.rmtree(results_dir)
            os.makedirs(results_dir)
        else:
            # se o arquivo rs_basic_search.json não foi encontrado, então apaga o diretório de resultados e reinicia a busca.
            print('ERRO. o arquivo rs_basic_search.json não foi encontrado no diretório de resultados. resetando o diretório.')
            shutil.rmtree(results_dir)
            os.makedirs(results_dir)
    else:
        os.makedirs(results_dir)

    create_rs_basic_search_json(results_dir)

    while True:
        rs_basic_search = load_rs_basic_search_json(results_dir)
        n_basic_experiments: int = rs_basic_search['n_basic_experiments']
        index = n_basic_experiments + 1

        if index > random_seed_max:
            print(f'nn_train_rs_basic_search: CONCLUÍDO. n_basic_experiments = {n_basic_experiments} == '
                  f'params_rs_search["random_seed_max"] ({random_seed_max})')
            break

        train_config = train_model(working_dir, settings, params_rs_search, seed=index, patience_style='short')

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

        basic_experiments = rs_basic_search['basic_experiments']
        rs_basic_search['n_basic_experiments'] = len(basic_experiments)
        sorted_basic_experiments = sorted(basic_experiments, key=lambda d: d['losses_product'])
        rs_basic_search['sorted_basic_experiments'] = sorted_basic_experiments

        update_rs_basic_search_json(results_dir, rs_basic_search)

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

    run_setup()
    nn_train_rs_basic_search()
