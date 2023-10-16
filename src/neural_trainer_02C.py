# faz um pesquisa mais profunda para encontrar o melhor random_seed.
# usa o arquivo train_log.json
# faz treinamentos com patience=15

import os
import copy
import time

from utils_filesystem import read_json, write_json, write_train_config
from neural_trainer_utils import train_model


params_nn = read_json('params_nn.json')
filename_deep = 'rs_deep_search.json'


def load_rs_deep_search_json() -> dict:
    if os.path.exists(filename_deep):
        _dict = read_json(filename_deep)
        return _dict
    else:
        print(f'ERRO. o arquivo {filename_deep} não existe.')
        exit(-1)


def nn_train_with_best_deep_random_seed():
    rs_deep_search = load_rs_deep_search_json()
    seed: int = rs_deep_search['best_deep_random_seed']

    settings = read_json('settings.json')
    settings['random_seed'] = seed
    write_json('settings.json', settings)

    train_config = train_model(settings, params_nn, seed, patience_style='long')
    write_train_config(train_config)

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

    print(_dict)

    log = {
        'random_seed': seed,
        'effective_n_epochs': train_config['effective_n_epochs'],
        'whole_set_train_loss': whole_set_train_loss_eval,
        'test_loss': test_loss_eval,
        'losses_product': losses_product
    }
    print(log)

    pass


if __name__ == '__main__':
    nn_train_with_best_deep_random_seed()
