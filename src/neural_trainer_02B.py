# faz um pesquisa mais profunda para encontrar o melhor random_seed.
# usa o arquivo train_log.json
# faz treinamentos com patience=15

import os
import time

from utils_filesystem import read_json, write_json
from neural_trainer_utils import train_model


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


def test_13():
    train_log = load_train_log()
    experiments = train_log['experiments']
    sorted_exps_by_test_loss = sorted(experiments, key=lambda d: d['test_loss'])
    sorted_exps_by_whole_set_train_loss = sorted(experiments, key=lambda d: d['whole_set_train_loss'])
    sorted_exps_by_product_losses = sorted(experiments, key=lambda d: d['product'])

    print('sorted_exps_by_product_losses:')
    print_list(sorted_exps_by_product_losses)

    print('sorted_exps_by_test_loss:')
    print_list(sorted_exps_by_test_loss)

    print('sorted_exps_by_whole_set_train_loss:')
    print_list(sorted_exps_by_whole_set_train_loss)


def trainer_01():
    create_train_log()
    _secs = 40

    while True:
        train_log = load_train_log()
        index = train_log['n_experiments'] + 1

        train_config = train_model(deterministic=True, seed=index)

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
    trainer_01()

# 5 anos de histórico
# TF(MIN)    PAUSA(S)
#  5        80
# 10        60
# 15        50
# 20        40
# 30        30
# 60        20
