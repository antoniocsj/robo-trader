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

import pickle
import json
from numpy import ndarray
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

from utils import NpEncoder
import time


def denorm_close_price(_c, trans: MinMaxScaler):
    c_denorm = trans.inverse_transform(np.array([0, 0, 0, _c, 0], dtype=object).reshape(1, -1))
    c_denorm = c_denorm[0][3]
    return c_denorm


def save_train_configs(_train_configs: dict):
    with open("train_configs.json", "w") as file:
        json.dump(_train_configs, file, indent=4, sort_keys=False, cls=NpEncoder)


# usada nas redes neurais
def prepare_train_data_multi(_hist: HistMulti, _symbol_out: str, _start_index: int,
                             _num_velas: int, _tipo_vela: str) -> ndarray:
    _data = []
    _timeframe = _hist.timeframe
    _symbol_tf_out = f'{_symbol_out}_{_timeframe}'

    if _tipo_vela == 'C':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            _c_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 5]
            _c_in = _c_in.reshape(len(_c_in), 1)
            if len(_data) == 0:
                _data = _c_in
            else:
                _data = np.hstack((_data, _c_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 5]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _tipo_vela == 'CV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            _cv_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 5:7]
            if len(_data) == 0:
                _data = _cv_in
            else:
                _data = np.hstack((_data, _cv_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 5]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _tipo_vela == 'OHLC':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            _ohlc_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 2:6]
            if len(_data) == 0:
                _data = _ohlc_in
            else:
                _data = np.hstack((_data, _ohlc_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 5]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _tipo_vela == 'OHLCV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            _ohlcv_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 2:7]
            if len(_data) == 0:
                _data = _ohlcv_in
            else:
                _data = np.hstack((_data, _ohlcv_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 5]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    else:
        print(f'tipo de vela não suportado: {_tipo_vela}')
        exit(-1)

    return _data


# split a multivariate sequence into samples.
# We can define a function named split_sequences() that will take a dataset as we have defined it with rows for
# time steps and columns for parallel series and return input/output samples.
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


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
    dir_csv = '../csv'
    hist = HistMulti(directory=dir_csv)

    symbol_out = 'EURUSD'
    n_steps = 2
    tipo_vela = 'OHLCV'
    n_samples_train = 30000
    validation_split = 0.1

    num_ativos = len(hist.symbols)
    num_entradas = num_ativos * n_steps * len(tipo_vela)
    max_n_epochs = num_entradas
    patience = int(max_n_epochs / 10)

    # horizontally stack columns
    dataset_train = prepare_train_data_multi(hist, symbol_out, 0, n_samples_train, tipo_vela)

    # convert into input/output
    X_train, y_train = split_sequences(dataset_train, n_steps)
    print(X_train.shape, y_train.shape)

    # We are now ready to fit a 1D CNN model on this data, specifying the expected number of time steps and
    # features to expect for each input sample, in this case three and two respectively.

    n_features = X_train.shape[2]

    # define model
    model = Sequential()
    model.add(Conv1D(filters=num_entradas, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(num_entradas, activation='relu'))
    model.add(Dense(num_entradas, activation='relu'))
    # model.add(Dense(num_entradas, activation='relu'))
    # model.add(Dense(num_entradas, activation='relu'))
    # model.add(Dense(num_entradas, activation='relu'))
    model.add(Dense(1))
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
    n_samples_test = 1500
    samples_index_start = n_samples_train
    dataset_test = prepare_train_data_multi(hist, symbol_out, samples_index_start, n_samples_test, tipo_vela)

    X_test, y_test = split_sequences(dataset_test, n_steps)
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    saved_model = load_model('model.h5')
    test_loss_eval = saved_model.evaluate(X_test, y_test, verbose=0)
    print(f'test_loss_eval: {test_loss_eval} (n_samples_test = {n_samples_test})')

    train_configs = {'symbol_out': symbol_out,
                     'tipo_vela': tipo_vela,
                     'timeframe': hist.timeframe,
                     'n_steps': n_steps,
                     'n_symbols': num_ativos,
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
                     'model_config': model_config,
                     'symbols': hist.symbols,
                     'history': history.history}

    save_train_configs(train_configs)


def evaluate_model():
    dir_csv = '../csv'
    hist = HistMulti(directory=dir_csv)

    # with open('train_configs.pkl', 'rb') as file:
    #     train_configs = pickle.load(file)
    with open("train_configs.json", "r") as file:
        train_configs = json.load(file)

    print(f'train_configs:')
    print(f'{train_configs}')

    n_steps = train_configs['n_steps']
    n_features = train_configs['n_features']
    symbol_out = train_configs['symbol_out']
    n_samples_train = train_configs['n_samples_train']
    tipo_vela = train_configs['tipo_vela']

    n_samples_test = 1000
    samples_index_start = n_samples_train
    dataset_test = prepare_train_data_multi(hist, symbol_out, samples_index_start, n_samples_test, tipo_vela)

    X_test, y_test = split_sequences(dataset_test, n_steps)
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    saved_model = load_model('model.h5')
    test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {test_acc} (n_samples_test = {n_samples_test})')


def calculate_model_bias():
    dir_csv = '../csv'
    hist = HistMulti(directory=dir_csv)

    with open("train_configs.json", "r") as file:
        train_configs = json.load(file)

    print(f'train_configs:')
    print(f'{train_configs}')

    n_steps = train_configs['n_steps']
    n_features = train_configs['n_features']
    symbol_out = train_configs['symbol_out']
    n_samples_train = train_configs['n_samples_train']
    tipo_vela = train_configs['tipo_vela']
    validation_split = train_configs['validation_split']
    istart_samples_test = int(n_samples_train * validation_split)
    n_samples_test = int(n_samples_train * validation_split)
    print(f'calculando o bias do modelo. (n_samples_test = {n_samples_test})')

    dataset_test = prepare_train_data_multi(hist, symbol_out, istart_samples_test, n_samples_test, tipo_vela)
    X_, y_ = split_sequences(dataset_test, n_steps)
    print(X_.shape, y_.shape)

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
        diff = y_[i] - y_pred[0][0]
        diffs.append(diff)
        if i % 3000 == 0 and i > 0:
            _t = 60
            print(f'esperando {_t} segundos para continuar')
            time.sleep(_t)

    diffs = np.asarray(diffs)
    bias = np.sum(diffs) / len(diffs)
    print(f'bias = {bias}')

    train_configs['bias'] = bias
    train_configs['n_samples_test_for_calc_bias'] = n_samples_test
    save_train_configs(train_configs)
    print('bias salvo em train_configs.json')


def test_model_with_trader():
    from TraderSimMultiNoPrints import TraderSimMulti

    dir_csv = '../csv'
    hist = HistMulti(directory=dir_csv)

    with open("train_configs.json", "r") as file:
        train_configs = json.load(file)

    print(f'train_configs:')
    print(f'{train_configs}')

    n_steps = train_configs['n_steps']
    n_features = train_configs['n_features']
    symbol_out = train_configs['symbol_out']
    n_samples_train = train_configs['n_samples_train']
    bias = train_configs['bias']
    tipo_vela = train_configs['tipo_vela']
    n_samples_test = 1500
    samples_index_start = n_samples_train

    dataset_test = prepare_train_data_multi(hist, symbol_out, samples_index_start, n_samples_test, tipo_vela)
    X_, y_ = split_sequences(dataset_test, n_steps)
    print(X_.shape, y_.shape)

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
    counter = 0
    for i in range(samples_index_start + n_steps, samples_index_start + candlesticks_quantity):
        print(f'i = {i}, {100 * (i-samples_index_start-n_steps) / (candlesticks_quantity - n_steps):.2f} %')
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
        # if close_pred_denorm > current_price:
        #     trader.buy(symbol_out)
        # else:
        #     trader.sell(symbol_out)

        if trader.profit >= 0.5 and trader.profit <= -0.5:
            trader.close_position()
        else:
            dif = close_pred_denorm - current_price
            if dif > 0 and abs(dif) > 0.00035:
                trader.buy(symbol_out)
            elif dif < 0 and abs(dif) > 0.00035:
                trader.sell(symbol_out)

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

    dir_csv = '../csv'
    hist = HistMulti(directory=dir_csv)

    with open("train_configs.json", "r") as file:
        train_configs = json.load(file)

    print(f'train_configs:')
    print(f'{train_configs}')

    n_steps = train_configs['n_steps']
    n_features = train_configs['n_features']
    symbol_out = train_configs['symbol_out']
    n_samples_train = train_configs['n_samples_train']
    bias = train_configs['bias']
    tipo_vela = train_configs['tipo_vela']
    n_samples_test = 1500
    samples_index_start = n_samples_train

    dataset_test = prepare_train_data_multi(hist, symbol_out, samples_index_start, n_samples_test, tipo_vela)
    X_, y_ = split_sequences(dataset_test, n_steps)
    print(X_.shape, y_.shape)

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
        print(f'i = {i}, {100 * (i-samples_index_start-n_steps) / (candlesticks_quantity - n_steps):.2f} %')
        trader.index = i
        trader.print_symbols_close_price_at(i)
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
        print(f'fechamento da vela atual {symbol_out}: {current_price:.5f}')
        print(f'previsão para o fechamento da próxima vela {symbol_out}: {close_pred_denorm:.5f} '
              f'(dif = {close_pred_denorm-current_price:.5f})')

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
    show_tf()
    # train_model()
    calculate_model_bias()
    # test_model_with_trader()
    # test_model_with_trader_interactive()
