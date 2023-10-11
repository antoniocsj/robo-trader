# executa vários treinamentos com random_seeds diferentes
# com patience=5 e guarda os resultados em train_log.json

import os
import time

from utils_filesystem import read_json, write_json
from neural_trainer_utils import train_model


params_nn = read_json('params_nn.json')


def create_train_log():
    import os
    _filename = 'train_log.json'
    if not os.path.exists(_filename):
        print(f'o arquivo {_filename} não existe ainda. será criado agora.')
        _dict = {
            'n_experiments': 0,
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
            'experiments': []
        }
        write_json(_filename, _dict)
    else:
        print(f'o arquivo {_filename} já existe. continuando os experimentos.')


def load_train_log() -> dict:
    _filename = 'train_log.json'
    if os.path.exists(_filename):
        _dict = read_json(_filename)
        return _dict
    else:
        print(f'ERRO. o arquivo {_filename} não existe.')
        exit(-1)


def update_train_log(train_log: dict):
    _filename = 'train_log.json'
    if os.path.exists(_filename):
        write_json(_filename, train_log)
    else:
        print(f'ERRO. o arquivo {_filename} não existe.')
        exit(-1)


def trainer_01():
    settings = read_json('settings.json')
    create_train_log()
    _secs = 40

    while True:
        train_log = load_train_log()
        index = train_log['n_experiments'] + 1

        train_config = train_model(settings, params_nn=params_nn, seed=index)

        whole_set_train_loss_eval = train_config['whole_set_train_loss_eval']
        test_loss_eval = train_config['test_loss_eval']
        product = whole_set_train_loss_eval * test_loss_eval

        train_log['n_experiments'] = index

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

        train_log.update(_dict)

        log = {
            'random_seed': index,
            'effective_n_epochs': train_config['effective_n_epochs'],
            'whole_set_train_loss': whole_set_train_loss_eval,
            'test_loss': test_loss_eval,
            'product': product
        }
        train_log['experiments'].append(log)

        update_train_log(train_log=train_log)

        print(f'esperando por {_secs} segundos')
        time.sleep(_secs)


if __name__ == '__main__':
    _settings = read_json('settings.json')
    _settings['timeframe'] = params_nn['timeframe']
    _settings['candle_input_type'] = params_nn['candle_input_type']
    write_json('settings.json', _settings)

    trainer_01()


# 5 anos de histórico
# TF(MIN)    PAUSA(S)
#  5        80
# 10        60
# 15        50
# 20        40
# 30        30
# 60        20
