from utils_filesystem import read_json, write_json
from neural_trainer_02A import nn_train_scan_random_seeds
from neural_trainer_02B import nn_train_search_best_random_seed

params_nn = read_json('params_nn.json')
filename_basic = 'rs_basic_search.json'


def main():
    _settings = read_json('settings.json')
    _settings['timeframe'] = params_nn['timeframe']
    _settings['candle_input_type'] = params_nn['candle_input_type']
    _settings['random_seed'] = 1
    write_json('settings.json', _settings)

    nn_train_scan_random_seeds()
    nn_train_search_best_random_seed()


if __name__ == '__main__':
    main()
