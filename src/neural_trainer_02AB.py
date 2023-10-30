from utils_filesystem import read_json, write_json
from neural_trainer_02A import nn_train_rs_basic_search
from neural_trainer_02B import nn_train_rs_deep_search

params_rs_search = read_json('params_rs_search.json')
filename_basic = 'rs_basic_search.json'


def main():
    _settings = read_json('settings.json')
    _settings['timeframe'] = params_rs_search['timeframe']
    _settings['candle_input_type'] = params_rs_search['candle_input_type']
    _settings['random_seed'] = 1
    write_json('settings.json', _settings)

    nn_train_rs_basic_search()
    nn_train_rs_deep_search()


if __name__ == '__main__':
    main()
