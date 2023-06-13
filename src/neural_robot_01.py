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
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model

tf.keras.utils.set_random_seed(1)


def denorm_close_price(_c, trans: MinMaxScaler):
    c_denorm = trans.inverse_transform(np.array([0, 0, 0, _c, 0], dtype=object).reshape(1, -1))
    c_denorm = c_denorm[0][3]
    return c_denorm


def save_history(_history: dict):
    with open("history.json", "w") as file:
        json.dump(_history, file)


# usada nas redes neurais
def prepare_train_data_multi(_hist: HistMulti, _symbol_out: str, _start_index: int,
                             _num_velas: int, _tipo_vela: str) -> ndarray:
    _data = []
    _timeframe = _hist.timeframe
    _symbol_tf_out = f'{_symbol_out}_{_timeframe}'

    if _tipo_vela == 'CV':
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
        pass
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
    num_ativos = len(hist.symbols)
    n_steps = 4
    tipo_vela = 'CV'
    num_entradas = num_ativos * n_steps * len(tipo_vela)
    symbol_out = 'EURUSD'
    n_samples_train = 30000  # quantidade de velas usadas no treinamento
    validation_split = 0.5
    n_epochs = 30

    # horizontally stack columns
    dataset_train = prepare_train_data_multi(hist, symbol_out, 0, n_samples_train, tipo_vela)

    # convert into input/output
    X, y = split_sequences(dataset_train, n_steps)
    print(X.shape, y.shape)

    # summarize the data
    # for i in range(len(X)):
    #     print(X[i], y[i])

    # We are now ready to fit a 1D CNN model on this data, specifying the expected number of time steps and
    # features to expect for each input sample, in this case three and two respectively.

    n_features = X.shape[2]

    # define model
    model = Sequential()
    model.add(Conv1D(filters=num_entradas, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(num_entradas, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)
    history = model.fit(X, y, epochs=n_epochs, verbose=1, validation_split=validation_split)
    save_history(history.history)

    last_loss = model.history.history['loss'][-1]
    last_val_loss = model.history.history['val_loss'][-1]
    print(f'loss = {last_loss}, val_loss = {last_val_loss}')

    model.save('model.hdf5')

    model_configs = {'tipo_vela': tipo_vela,
                     'symbol_out': symbol_out,
                     'symbols': hist.symbols,
                     'n_steps': n_steps,
                     'n_features': n_features,
                     'n_samples_train': n_samples_train,
                     'validation_split': validation_split,
                     'n_epochs': n_epochs,
                     'num_entradas': num_entradas,
                     'last_loss': last_loss,
                     'last_val_loss': last_val_loss}

    with open('train_configs.pkl', 'wb') as file:
        pickle.dump(model_configs, file)

    print('treinamento concluído.')


def test_model():
    dir_csv = '../csv'
    hist = HistMulti(directory=dir_csv)

    with open('train_configs.pkl', 'rb') as file:
        train_configs = pickle.load(file)

    print(f'train_configs:')
    print(f'{train_configs}')

    n_steps = train_configs['n_steps']
    n_features = train_configs['n_features']
    symbol_out = train_configs['symbol_out']
    n_samples_train = train_configs['n_samples_train']
    tipo_vela = train_configs['tipo_vela']
    n_samples_test = 100

    dataset_test = prepare_train_data_multi(hist, symbol_out, n_samples_train, n_samples_test, tipo_vela)
    X_, y_ = split_sequences(dataset_test, n_steps)
    print(X_.shape, y_.shape)

    model = load_model('model.hdf5')

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)
    _symbol_timeframe = f'{symbol_out}_{hist.timeframe}'
    trans: MinMaxScaler = scalers[_symbol_timeframe]

    # demonstrate prediction
    # x_input = np.array([[80, 85], [90, 95], [100, 105]])
    # x_input = x_input.reshape((1, n_steps, n_features))
    X_ = np.asarray(X_).astype(np.float32)
    y_ = np.asarray(y_).astype(np.float32)

    for i in range(50):
        x_input = X_[i]
        x_input = x_input.reshape((1, n_steps, n_features))
        y_pred_norm = model.predict(x_input)
        y_denorm = denorm_close_price(y_[i], trans)
        y_pred_denorm = denorm_close_price(y_pred_norm[0][0], trans)
        diff_real = y_pred_denorm - y_denorm
        print(f'previsto = {y_pred_denorm}, real = {y_denorm}, dif = {diff_real}')


def test_model_with_trader():
    from TraderSimMultiNoPrints import TraderSimMulti

    dir_csv = '../csv'
    hist = HistMulti(directory=dir_csv)

    with open('train_configs.pkl', 'rb') as file:
        train_configs = pickle.load(file)

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
    X_, y_ = split_sequences(dataset_test, n_steps)
    print(X_.shape, y_.shape)

    model = load_model('model.hdf5')

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)
    _symbol_timeframe = f'{symbol_out}_{hist.timeframe}'
    trans: MinMaxScaler = scalers[_symbol_timeframe]

    # demonstrate prediction
    # x_input = np.array([[80, 85], [90, 95], [100, 105]])
    # x_input = x_input.reshape((1, n_steps, n_features))
    X_ = np.asarray(X_).astype(np.float32)
    y_ = np.asarray(y_).astype(np.float32)

    initial_deposit = 1000.0

    trader = TraderSimMulti(initial_deposit)
    trader.max_candlestick_count = 5
    trader.start_simulation()

    candlesticks_quantity = n_samples_test  # quantidade de velas que serão usadas na simulação

    for i in range(samples_index_start + n_steps, samples_index_start + candlesticks_quantity):
        # print(f'i = {i}')
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
        # close_denorm = denorm_close_price(y_[j], trans)
        close_pred_denorm = denorm_close_price(close_pred_norm[0][0], trans)

        # aqui toma-se a decisão de comprar ou vender baseando-se no valor da previsão
        if close_pred_denorm > current_price:
            trader.buy(symbol_out)
        else:
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

        # trader.print_trade_stats()

    print('\nresultados finais da simulação')
    trader.print_trade_stats()


def show_tf():
    print(os.environ["LD_LIBRARY_PATH"])
    print(os.environ["PYTHONPATH"])
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))


if __name__ == '__main__':
    show_tf()
    train_model()
    # test_model()
    # test_model_with_trader()
