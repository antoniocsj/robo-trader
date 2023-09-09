import numpy as np

from HistMulti import HistMulti
from Predictor import Predictor
from utils_ops import denorm_output
from utils_nn import split_sequences2, prepare_train_data3


def evaluate_predictor(pred: Predictor):
    settings = pred.settings
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    timeframe = settings['timeframe']

    train_config = pred.train_config
    print(f'train_config:')
    print(f'{train_config}')

    hist = HistMulti(temp_dir, timeframe, symbols_allowed=train_config['symbols'])

    n_steps = train_config['n_steps']
    n_features = train_config['n_features']
    symbol_out = train_config['symbol_out']
    timeframe = train_config['timeframe']
    symbol_tf = f'{symbol_out}_{timeframe}'
    n_samples_train = train_config['n_samples_train']
    # bias = train_config['bias']
    bias = 0.0
    candle_input_type = train_config['candle_input_type']
    candle_output_type = train_config['candle_output_type']
    n_samples_test = 3000
    samples_index_start = n_samples_train

    dataset_test = prepare_train_data3(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type,
                                       candle_output_type)
    X, y = split_sequences2(dataset_test, n_steps, candle_output_type)

    dataset_test2 = prepare_train_data3(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type,
                                        'HL')
    X2, y_hl = split_sequences2(dataset_test2, n_steps, 'HL')

    model = pred.model
    scalers = pred.scalers
    scaler = scalers[symbol_tf]

    candlesticks_quantity = len(X)  # quantidade de velas que serão usadas na simulação
    num_hits = 0

    for i in range(candlesticks_quantity):
        print(f'i = {i}, {100 * i / candlesticks_quantity:.2f} %')

        # aqui a rede neural faz a previsão do valor de fechamento da próxima vela
        x_input = X[i]
        x_input = x_input.reshape((1, n_steps, n_features))

        y_norm = np.array(y[i], ndmin=2)
        y_denorm = denorm_output(y_norm, 0.0, candle_output_type, scaler)

        y_pred_norm = model.predict(x_input)
        y_pred_denorm = denorm_output(y_pred_norm, bias, candle_output_type, scaler)

        hl = np.array(y_hl[i], ndmin=2)
        hl_denorm = denorm_output(hl, 0.0, 'HL', scaler)
        _H, _L = hl_denorm[1], hl_denorm[2]
        # print(f'y = {y_denorm:.5f}, y_pred = {y_pred_denorm:.5f}, HL = [{_H:.5f} {_L:.5f}]')
        # pred.calc_output(input_data)
        # output_denorm2 = pred.output

        if _L < y_pred_denorm < _H:
            num_hits += 1

    hit_rate = num_hits / candlesticks_quantity
    print(f'hit_rate = {100 * hit_rate:.2f} %')

    return hit_rate


if __name__ == '__main__':
    hit_rates = []

    predictor = Predictor(1)
    _hit_rate = evaluate_predictor(predictor)
    hit_rates.append(_hit_rate)

    predictor = Predictor(2)
    _hit_rate = evaluate_predictor(predictor)
    hit_rates.append(_hit_rate)

    predictor = Predictor(3)
    _hit_rate = evaluate_predictor(predictor)
    hit_rates.append(_hit_rate)

    predictor = Predictor(4)
    _hit_rate = evaluate_predictor(predictor)
    hit_rates.append(_hit_rate)

    predictor = Predictor(5)
    _hit_rate = evaluate_predictor(predictor)
    hit_rates.append(_hit_rate)

    for i in range(len(hit_rates)):
        print(f'predictor {i+1}: {hit_rates[i]:.2f} %')

    average = sum(hit_rates) / len(hit_rates)
    print(f'average hit_rate = {average:.2f} %')
