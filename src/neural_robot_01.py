import os

os.environ['PYTHONHASHSEED'] = str(1)
os.environ['TF_CUDNN_DETERMINISM'] = str(1)
os.environ['TF_DETERMINISTIC_OPS'] = str(1)

import numpy as np

np.random.seed(1)

import random

random.seed(1)

import tensorflow as tf

tf.random.set_seed(1)
tf.keras.utils.set_random_seed(1)

import time
import pickle
import json
from HistMulti import HistMulti
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.constraints import MaxNorm
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

tf.keras.utils.set_random_seed(1)

from utils_nn import prepare_train_data, split_sequences1, split_sequences2, prepare_train_data2
from utils_filesystem import read_json, save_train_configs
from utils_ops import denorm_close_price
from utils_symbols import calc_n_inputs


# Multivariate CNN Models
# Multivariate time series data means data where there is more than one observation for each time step.
#
# There are two main models that we may require with multivariate time series data; they are:
#
# Multiple Input Series.
# Multiple Parallel Series.
# Let’s take a look at each in turn.
#
# Multiple Input Series
# A problem may have two or more parallel input time series and an output time series that is dependent
# on the input time series.
#
# The input time series are parallel because each series has observations at the same time steps.
#
# We can demonstrate this with a simple example of two parallel input time series where the output series
# is the simple addition of the input series.
def train_model():
    setup = read_json('setup.json')
    print(f'setup.json: {setup}')

    csv_dir = setup['csv_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']
    candle_input_type = setup['candle_input_type']
    candle_output_type = setup['candle_output_type']
    hist = HistMulti(csv_dir, timeframe)

    if hist.timeframe != timeframe:
        print(f'o timeframe do diretório {csv_dir} ({hist.timeframe}) é diferente do timeframe especificado '
              f'em setup.json ({timeframe})')
        exit(-1)

    n_steps = 2
    n_samples_train = 500  # 30000-M10, 60000-M5
    validation_split = 0.5

    n_cols, n_symbols = calc_n_inputs(csv_dir, candle_input_type, timeframe)
    num_entradas = n_steps * n_cols
    max_n_epochs = num_entradas
    patience = int(max_n_epochs / 10)

    print(f'n_steps = {n_steps}, tipo_vela_entrada = {candle_input_type}, tipo_vela_saída = {candle_output_type}, '
          f'n_samples_train = {n_samples_train}, validation_split = {validation_split}, max_n_epochs = {max_n_epochs}, '
          f'patience = {patience}')

    # horizontally stack columns
    # dataset_train = prepare_train_data(hist, symbol_out, 0, n_samples_train, candle_input_type)
    dataset_train = prepare_train_data2(hist, symbol_out, 0, n_samples_train, candle_input_type,
                                        candle_output_type)

    # convert into input/output samples
    # X_train, y_train = split_sequences1(dataset_train, n_steps)
    X_train, y_train = split_sequences2(dataset_train, n_steps, candle_output_type)

    # We are now ready to fit a 1D CNN model on this data, specifying the expected number of time steps and
    # features to expect for each input sample.
    n_features = X_train.shape[2]

    # define model
    model = Sequential()
    model.add(Conv1D(filters=n_features, kernel_size=n_steps, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(n_features, activation='relu'))
    model.add(Dense(n_features, activation='relu'))
    model.add(Dense(len(candle_output_type)))
    model.compile(optimizer='adam', loss='mse')
    model_config = model.get_config()

    # fit model
    print(f'treinando o modelo em parte das amostras de treinamento.')
    print(f'n_samples_train * validation_split = {n_samples_train} * {validation_split} = '
          f'{int(n_samples_train * validation_split)}).')
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
                 ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True, verbose=1)]
    history = model.fit(X_train, y_train, epochs=max_n_epochs, verbose=1,
                        validation_split=validation_split, callbacks=callbacks)

    effective_n_epochs = len(history.history['loss'])
    loss, val_loss = history.history['loss'], history.history['val_loss']
    i_min_loss, i_min_val_loss = np.argmin(loss), np.argmin(val_loss)
    min_loss, min_val_loss = loss[i_min_loss], val_loss[i_min_val_loss]
    losses = {'min_loss': {'value': min_loss, 'index': i_min_loss, 'epoch': i_min_loss + 1},
              'min_val_loss': {'value': min_val_loss, 'index': i_min_val_loss, 'epoch': i_min_val_loss + 1}}

    print(f'avaliando o modelo no conjunto inteiro das amostras de treinamento.')
    saved_model = load_model('model.h5')
    whole_set_train_loss_eval = saved_model.evaluate(X_train, y_train, verbose=0)
    print(f'whole_set_train_loss_eval: {whole_set_train_loss_eval:} (n_samples_train = {n_samples_train})')

    print(f'avaliando o modelo num novo conjunto de amostras de teste.')
    n_samples_test = 500
    samples_index_start = n_samples_train
    # dataset_test = prepare_train_data(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type)
    dataset_test = prepare_train_data2(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type,
                                       candle_output_type)

    # X_test, y_test = split_sequences1(dataset_test, n_steps)
    X_test, y_test = split_sequences2(dataset_test, n_steps, candle_output_type)
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    saved_model = load_model('model.h5')
    test_loss_eval = saved_model.evaluate(X_test, y_test, verbose=0)
    print(f'test_loss_eval: {test_loss_eval} (n_samples_test = {n_samples_test})')

    train_configs = {'symbol_out': symbol_out,
                     'timeframe': hist.timeframe,
                     'n_steps': n_steps,
                     'candle_input_type': candle_input_type,
                     'candle_output_type': candle_output_type,
                     'n_symbols': n_symbols,
                     'n_features': n_features,
                     'num_entradas': num_entradas,
                     'n_samples_train': n_samples_train,
                     'validation_split': validation_split,
                     'effective_n_epochs': effective_n_epochs,
                     'max_n_epochs': max_n_epochs,
                     'patience': patience,
                     'whole_set_train_loss_eval': whole_set_train_loss_eval,
                     'n_samples_test': n_samples_test,
                     'test_loss_eval': test_loss_eval,
                     'losses': losses,
                     'symbols': hist.symbols,
                     'bias': 0.0,
                     'n_samples_test_for_calc_bias': 0,
                     'model_config': model_config,
                     'history': history.history}

    save_train_configs(train_configs)


def train_model_return(setup: dict, hist: HistMulti, n_steps: int, layer_type: list):
    csv_dir = setup['csv_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']
    candle_input_type = setup['candle_input_type']
    candle_output_type = setup['candle_output_type']
    hist = hist

    n_steps = n_steps
    n_samples_train = 1000  # 30000-M10, 60000-M5
    validation_split = 0.2

    n_cols, n_symbols = calc_n_inputs(csv_dir, candle_input_type, timeframe)
    num_entradas = n_steps * n_cols
    n_epochs = 2

    print(f'n_steps = {n_steps}, tipo_vela_entrada = {candle_input_type}, tipo_vela_saída = {candle_output_type}, '
          f'n_samples_train = {n_samples_train}, validation_split = {validation_split}, n_epochs = {n_epochs} '
          f'layer_type = {layer_type}')

    # horizontally stack columns
    dataset_train = prepare_train_data(hist, symbol_out, 0, n_samples_train, candle_input_type)

    # convert into input/output
    X_train, y_train = split_sequences1(dataset_train, n_steps)
    print(X_train.shape, y_train.shape)

    # We are now ready to fit a 1D CNN model on this data, specifying the expected number of time steps and
    # features to expect for each input sample.

    n_features = X_train.shape[2]

    # define model
    model = Sequential()
    model.add(Conv1D(filters=n_features, kernel_size=n_steps, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Flatten())

    if sum(layer_type) > 0:
        for i in range(len(layer_type)):
            if layer_type[i] != 0:
                model.add(Dense(layer_type[i], activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    model.fit(X_train, y_train, epochs=n_epochs, verbose=0, validation_split=validation_split)

    whole_set_train_loss_eval = model.evaluate(X_train, y_train, verbose=0)
    print(f'whole_set_train_loss_eval: {whole_set_train_loss_eval:} (n_samples_train = {n_samples_train})')

    n_samples_test = 1000
    samples_index_start = n_samples_train
    dataset_test = prepare_train_data(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type)

    X_test, y_test = split_sequences1(dataset_test, n_steps)
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    test_loss_eval = model.evaluate(X_test, y_test, verbose=0)
    print(f'test_loss_eval: {test_loss_eval} (n_samples_test = {n_samples_test})')

    train_configs = {'symbol_out': symbol_out,
                     'timeframe': hist.timeframe,
                     'n_steps': n_steps,
                     'candle_input_type': candle_input_type,
                     'candle_output_type': candle_output_type,
                     'n_symbols': n_symbols,
                     'n_features': n_features,
                     'num_entradas': num_entradas,
                     'n_samples_train': n_samples_train,
                     'validation_split': validation_split,
                     'n_epochs': n_epochs,
                     'layer_type': list(layer_type),
                     'whole_set_train_loss_eval': whole_set_train_loss_eval,
                     'n_samples_test': n_samples_test,
                     'test_loss_eval': test_loss_eval,
                     'symbols': hist.symbols}

    return train_configs


def test_models():
    setup = read_json('setup.json')
    print(f'setup.json: {setup}')

    csv_dir = setup['csv_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']
    candle_input_type = setup['candle_input_type']
    candle_output_type = setup['candle_output_type']

    hist = HistMulti(csv_dir, timeframe)
    n_steps = 2
    n_cols, n_symbols = calc_n_inputs(csv_dir, candle_input_type, timeframe)
    num_entradas = n_steps * n_cols

    import itertools as it

    min_n_neurons = 0
    max_n_neurons = num_entradas
    step = 30
    _list_n_neurons = list(range(min_n_neurons, max_n_neurons + 1, step))
    n_layers = 3
    layers_comb = list(it.combinations_with_replacement(_list_n_neurons, n_layers))

    if os.path.exists('test_models.json'):
        with open('test_models.json', 'r') as file:
            _list_train_configs = json.load(file)
            i_start = len(_list_train_configs)
    else:
        _list_train_configs = []
        i_start = 0

    max_n_neurons = num_entradas
    wait = 1
    _len_layers_comb = len(layers_comb)
    for i in range(i_start, len(layers_comb)):
        _layer_type = sorted(list(layers_comb[i]), reverse=True)
        print(f'testando modelo com _layer_type = {_layer_type}, len_layers_comb = {_len_layers_comb}. '
              f'({100 * i / _len_layers_comb:.2f} %)')
        _out = train_model_return(setup, hist, n_steps, _layer_type)

        _list_train_configs.append(_out)
        with open("test_models.json", "w") as file:
            json.dump(_list_train_configs, file, indent=4)

        print(f'esperando {wait} segundos')
        time.sleep(wait)


def evaluate_model():
    setup = read_json('setup.json')
    print(f'setup.json: {setup}')

    csv_dir = setup['csv_dir']
    timeframe = setup['timeframe']
    hist = HistMulti(csv_dir, timeframe)

    with open("train_configs.json", "r") as file:
        train_configs = json.load(file)

    print(f'train_configs:')
    print(f'{train_configs}')

    n_steps = train_configs['n_steps']
    n_features = train_configs['n_features']
    symbol_out = train_configs['symbol_out']
    n_samples_train = train_configs['n_samples_train']
    candle_input_type = train_configs['candle_input_type']

    n_samples_test = 500
    samples_index_start = n_samples_train
    dataset_test = prepare_train_data(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type)

    X_test, y_test = split_sequences1(dataset_test, n_steps)
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    saved_model = load_model('model.h5')
    test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {test_acc} (n_samples_test = {n_samples_test})')


def calculate_model_bias():
    setup = read_json('setup.json')
    print(f'setup.json: {setup}')

    csv_dir = setup['csv_dir']
    timeframe = setup['timeframe']
    hist = HistMulti(csv_dir, timeframe)

    with open("train_configs.json", "r") as file:
        train_configs = json.load(file)

    print(f'train_configs:')
    print(f'{train_configs}')

    n_steps = train_configs['n_steps']
    n_features = train_configs['n_features']
    symbol_out = train_configs['symbol_out']
    n_samples_train = train_configs['n_samples_train']
    candle_input_type = train_configs['candle_input_type']
    candle_output_type = train_configs['candle_output_type']
    validation_split = train_configs['validation_split']
    istart_samples_test = int(n_samples_train * validation_split)
    n_samples_test = int(n_samples_train * validation_split)
    print(f'calculando o bias do modelo. (n_samples_test = {n_samples_test})')

    # dataset_test = prepare_train_data(hist, symbol_out, istart_samples_test, n_samples_test, candle_input_type)
    dataset_test = prepare_train_data2(hist, symbol_out, istart_samples_test, n_samples_test, candle_input_type,
                                       candle_output_type)

    # X_, y_ = split_sequences1(dataset_test, n_steps)
    X_, y_ = split_sequences2(dataset_test, n_steps, candle_output_type)

    model = load_model('model.h5')

    X_ = np.asarray(X_).astype(np.float32)
    y_ = np.asarray(y_).astype(np.float32)

    diffs = []
    len_X_ = len(X_)
    for i in range(len_X_):
        print(f'{100 * i / len_X_:.2f} %')
        x_input = X_[i]
        x_input = x_input.reshape((1, n_steps, n_features))
        y_pred = model.predict(x_input)
        if len(y_pred[0]) == 1:
            diff = y_[i] - y_pred[0][0]
        else:
            diff = y_[i] - y_pred[0]
        diffs.append(diff)
        if i % 3000 == 0 and i > 0:
            _t = 60
            print(f'esperando {_t} segundos para continuar')
            time.sleep(_t)

    diffs = np.asarray(diffs)
    bias = np.mean(diffs, axis=0)
    print(f'bias = {bias}')

    train_configs['bias'] = bias
    train_configs['n_samples_test_for_calc_bias'] = n_samples_test
    save_train_configs(train_configs)
    print('bias salvo em train_configs.json')


def test_model_with_trader():
    from TraderSimMultiNoPrints import TraderSimMulti

    setup = read_json('setup.json')
    print(f'setup.json: {setup}')

    csv_dir = setup['csv_dir']
    timeframe = setup['timeframe']
    hist = HistMulti(csv_dir, timeframe)

    with open("train_configs.json", "r") as file:
        train_configs = json.load(file)

    print(f'train_configs:')
    print(f'{train_configs}')

    n_steps = train_configs['n_steps']
    n_features = train_configs['n_features']
    symbol_out = train_configs['symbol_out']
    n_samples_train = train_configs['n_samples_train']
    bias = train_configs['bias']
    candle_input_type = train_configs['candle_input_type']
    n_samples_test = 500
    samples_index_start = n_samples_train

    dataset_test = prepare_train_data(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type)
    X_, y_ = split_sequences1(dataset_test, n_steps)

    model = load_model('model.h5')

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    _symbol_timeframe = f'{symbol_out}_{hist.timeframe}'
    trans: MinMaxScaler = scalers[_symbol_timeframe]

    X_ = np.asarray(X_).astype(np.float32)
    # y_ = np.asarray(y_).astype(np.float32)

    initial_deposit = 1000.0

    trader = TraderSimMulti(initial_deposit)
    trader.max_candlestick_count = 1
    trader.start_simulation()

    candlesticks_quantity = n_samples_test  # quantidade de velas que serão usadas na simulação
    counter = 0
    for i in range(samples_index_start + n_steps, samples_index_start + candlesticks_quantity):
        print(f'i = {i}, {100 * (i - samples_index_start - n_steps) / (candlesticks_quantity - n_steps):.2f} %')
        trader.index = i
        # trader.print_symbols_close_price_at(i)
        # trader.print_symbols_close_price_at(i, use_scalers=False)
        trader.update_profit()

        # fechamento da vela atual
        current_price = trader.get_close_price_symbol_at(symbol_out, i)

        # aqui a rede neural faz a previsão do valor de fechamento da próxima vela
        j = i - samples_index_start - n_steps + 1
        x_input = X_[j]
        x_input = x_input.reshape((1, n_steps, n_features))
        close_pred_norm = model.predict(x_input)
        close_pred_denorm = denorm_close_price(close_pred_norm[0][0] + bias, trans)

        # aqui toma-se a decisão de comprar ou vender baseando-se no valor da previsão
        if close_pred_denorm > current_price:
            trader.buy(symbol_out)
        else:
            trader.sell(symbol_out)

        # _tp = 0.5
        # _sl = 0.5
        # _k = 0.00020
        # if _tp <= trader.profit <= -_sl:
        #     trader.close_position()
        # else:
        #     dif = close_pred_denorm - current_price
        #     if dif > 0 and abs(dif) > _k:
        #         trader.buy(symbol_out)
        #     elif dif < 0 and abs(dif) > _k:
        #         trader.sell(symbol_out)

        # if _tp <= trader.profit <= -_sl:
        #     trader.close_position()
        #
        # dif = close_pred_denorm - current_price
        # if dif > 0 and abs(dif) > _k:
        #     trader.buy(symbol_out)
        # elif dif < 0 and abs(dif) > _k:
        #     trader.sell(symbol_out)

        if trader.profit < 0 and abs(trader.profit) / trader.balance >= trader.stop_loss:
            print(f'o stop_loss de {100 * trader.stop_loss:.2f} % for atingido.')
            trader.close_position()

        if trader.equity <= 0.0:
            trader.close_position()
            trader.finish_simulation()
            print('equity <= 0. a simulação será encerrada.')
            break

        # fecha a posição quando acabarem as novas barras (velas ou candlesticks)
        if i == samples_index_start + candlesticks_quantity - 1:
            trader.close_position()
            trader.finish_simulation()
            print('a última vela atingida. a simulação chegou ao fim.')

        if trader.candlestick_count >= trader.max_candlestick_count:
            print(f'fechamento forçado de negociações abertas. a contagem de velas atingiu o limite.')
            trader.close_position()

        if trader.open_position[0]:
            trader.candlestick_count += 1
        else:
            trader.candlestick_count = 0

        counter += 1
        if counter % 2000 == 0:
            time.sleep(60)

        # trader.print_trade_stats()

    print('\nresultados finais da simulação')
    print(f'n_samples_test = {n_samples_test}, max_candlestick_count = {trader.max_candlestick_count}')
    trader.print_trade_stats()


def test_model_with_trader_interactive():
    from TraderSimMultiNoPrints import TraderSimMulti

    setup = read_json('setup.json')
    print(f'setup.json: {setup}')

    csv_dir = setup['csv_dir']
    timeframe = setup['timeframe']
    hist = HistMulti(csv_dir, timeframe)

    with open("train_configs.json", "r") as file:
        train_configs = json.load(file)

    print(f'train_configs:')
    print(f'{train_configs}')

    n_steps = train_configs['n_steps']
    n_features = train_configs['n_features']
    symbol_out = train_configs['symbol_out']
    n_samples_train = train_configs['n_samples_train']
    bias = train_configs['bias']
    candle_input_type = train_configs['candle_input_type']
    n_samples_test = 500
    samples_index_start = n_samples_train

    dataset_test = prepare_train_data(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type)
    X_, y_ = split_sequences1(dataset_test, n_steps)

    model = load_model('model.h5')

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    _symbol_timeframe = f'{symbol_out}_{hist.timeframe}'
    trans: MinMaxScaler = scalers[_symbol_timeframe]

    X_ = np.asarray(X_).astype(np.float32)
    # y_ = np.asarray(y_).astype(np.float32)

    initial_deposit = 1000.0

    trader = TraderSimMulti(initial_deposit)
    trader.max_candlestick_count = 1000
    trader.start_simulation()

    candlesticks_quantity = n_samples_test  # quantidade de velas que serão usadas na simulação

    for i in range(samples_index_start + n_steps, samples_index_start + candlesticks_quantity):
        print(f'i = {i}, {100 * (i - samples_index_start - n_steps) / (candlesticks_quantity - n_steps):.2f} %')
        trader.index = i
        trader.update_profit()

        # fechamento da vela atual
        current_price = trader.get_close_price_symbol_at(symbol_out, i)

        # aqui a rede neural faz a previsão do valor de fechamento da próxima vela
        j = i - samples_index_start - n_steps + 1
        x_input = X_[j]
        x_input = x_input.reshape((1, n_steps, n_features))
        close_pred_norm = model.predict(x_input)
        close_pred_denorm = denorm_close_price(close_pred_norm[0][0] + bias, trans)
        print(f'fechamento da vela atual {symbol_out}: {current_price:.5f}')
        print(f'previsão para o fechamento da próxima vela {symbol_out}: {close_pred_denorm:.5f} '
              f'(dif = {close_pred_denorm - current_price:.5f})')

        if trader.profit < 0 and abs(trader.profit) / trader.balance >= trader.stop_loss:
            print(f'o stop_loss de {100 * trader.stop_loss:.2f} % for atingido.')
            trader.close_position()

        if trader.equity <= 0.0:
            trader.close_position()
            trader.finish_simulation()
            print('equity <= 0. a simulação será encerrada.')
            break

        # fecha a posição quando acabarem as novas barras (velas ou candlesticks)
        if i == samples_index_start + candlesticks_quantity - 1:
            trader.close_position()
            trader.finish_simulation()
            print('a última vela atingida. a simulação chegou ao fim.')

        if trader.candlestick_count >= trader.max_candlestick_count:
            print(f'fechamento forçado de negociações abertas. a contagem de velas atingiu o limite.')
            trader.close_position()

        if trader.open_position[0]:
            trader.candlestick_count += 1
        else:
            trader.candlestick_count = 0

        trader.print_trade_stats()

        ret_msg = trader.interact_with_user()
        if ret_msg == 'break':
            print('o usuário decidiu encerrar a simulação.')
            trader.close_position()
            trader.finish_simulation()
            break

    print('\nresultados finais da simulação')
    print(f'n_samples_test = {n_samples_test}, max_candlestick_count = {trader.max_candlestick_count}')
    trader.print_trade_stats()


def show_tf():
    print(os.environ["LD_LIBRARY_PATH"])
    print(os.environ["PYTHONPATH"])
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))


if __name__ == '__main__':
    # show_tf()
    # test_models()
    train_model()
    # calculate_model_bias()
    # test_model_with_trader()
    # test_model_with_trader_interactive()
