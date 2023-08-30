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
from HistMulti import HistMulti
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

tf.keras.utils.set_random_seed(1)

from utils_nn import prepare_train_data, split_sequences2, prepare_train_data2
from utils_filesystem import read_json, write_train_config, read_train_config
from utils_ops import denorm_close_price


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
    settings = read_json('settings.json')
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    candle_input_type = settings['candle_input_type']
    candle_output_type = settings['candle_output_type']
    hist = HistMulti(temp_dir, timeframe)

    if hist.timeframe != timeframe:
        print(f'o timeframe do diretório {temp_dir} ({hist.timeframe}) é diferente do timeframe especificado '
              f'em settings.json ({timeframe})')
        exit(-1)

    n_steps = 2
    n_samples_train = 72000  # 30000-M10, 72000-M5 Número de amostras usadas na fase de treinamento e validação
    validation_split = 0.2
    n_samples_test = 3000  # Número de amostras usadas na fase de avaliação. São amostras inéditas.
    # horizontally stack columns
    dataset_train = prepare_train_data2(hist, symbol_out, 0, n_samples_train, candle_input_type, candle_output_type)

    # convert into input/output samples
    X_train, y_train = split_sequences2(dataset_train, n_steps, candle_output_type)

    # We are now ready to fit a 1D CNN model on this data, specifying the expected number of time steps and
    # features to expect for each input sample.
    n_features = X_train.shape[2]
    n_inputs = n_steps * n_features
    max_n_epochs = n_inputs * 3 * 0 + 300
    patience = int(max_n_epochs / 10) * 0 + 40
    n_symbols = len(hist.symbols)

    print(f'symbols = {hist.symbols}')
    print(f'n_symbols = {n_symbols}, n_features (n_cols) = {n_features}, n_steps = {n_steps}, '
          f'tipo_vela_entrada = {candle_input_type}, tipo_vela_saída = {candle_output_type}, \n'
          f'n_samples_train = {n_samples_train}, validation_split = {validation_split}, '
          f'max_n_epochs = {max_n_epochs}, patience = {patience}')

    model = Sequential()

    # define cnn model
    # model.add(Conv1D(filters=1024, kernel_size=n_steps, activation='relu', input_shape=(n_steps, n_features)))
    # model.add(MaxPooling1D(pool_size=n_steps, padding='same'))
    # model.add(Flatten())
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(1024, activation='relu'))

    # define MLP model
    n_input = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], n_input))
    model.add(Dense(50, activation='sigmoid', input_dim=n_input))
    model.add(Dense(50, activation='sigmoid'))

    model.add(Dense(len(candle_output_type)))
    model.compile(optimizer='adam', loss='mse')
    model_config = model.get_config()

    # fit model
    print(f'treinando o modelo em parte das amostras de treinamento.')
    print(f'n_samples_train * validation_split = {n_samples_train} * {validation_split} = '
          f'{int(n_samples_train * validation_split)}).')

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
    samples_index_start = n_samples_train
    dataset_test = prepare_train_data2(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type,
                                       candle_output_type)

    X_test, y_test = split_sequences2(dataset_test, n_steps, candle_output_type)

    # for MLP model only
    n_input = X_test.shape[1] * X_test.shape[2]
    X_test = X_test.reshape((X_test.shape[0], n_input))

    saved_model = load_model('model.h5')
    test_loss_eval = saved_model.evaluate(X_test, y_test, verbose=0)
    print(f'test_loss_eval: {test_loss_eval} (n_samples_test = {n_samples_test})')

    train_config = {'symbol_out': symbol_out,
                    'timeframe': hist.timeframe,
                    'n_steps': n_steps,
                    'candle_input_type': candle_input_type,
                    'candle_output_type': candle_output_type,
                    'n_symbols': n_symbols,
                    'n_features': n_features,
                    'n_inputs': n_inputs,
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

    write_train_config(train_config)


def evaluate_model():
    settings = read_json('settings.json')
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    timeframe = settings['timeframe']
    hist = HistMulti(temp_dir, timeframe)

    train_config = read_train_config()

    print(f'train_config:')
    print(f'{train_config}')

    n_steps = train_config['n_steps']
    n_features = train_config['n_features']
    symbol_out = train_config['symbol_out']
    n_samples_train = train_config['n_samples_train']
    candle_input_type = train_config['candle_input_type']
    candle_output_type = settings['candle_output_type']

    n_samples_test = 100
    samples_index_start = n_samples_train
    dataset_test = prepare_train_data2(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type,
                                       candle_output_type)

    X_test, y_test = split_sequences2(dataset_test, n_steps, candle_output_type)

    saved_model = load_model('model.h5')
    test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {test_acc} (n_samples_test = {n_samples_test})')


def calculate_model_bias():
    settings = read_json('settings.json')
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    timeframe = settings['timeframe']
    hist = HistMulti(temp_dir, timeframe)

    train_config = read_train_config()

    print(f'train_config:')
    print(f'{train_config}')

    n_steps = train_config['n_steps']
    n_features = train_config['n_features']
    symbol_out = train_config['symbol_out']
    n_samples_train = train_config['n_samples_train']
    candle_input_type = train_config['candle_input_type']
    candle_output_type = train_config['candle_output_type']
    validation_split = train_config['validation_split']
    istart_samples_test = int(n_samples_train * validation_split)
    n_samples_test = int(n_samples_train * validation_split)

    print(f'calculando o bias do modelo. (n_samples_test = {n_samples_test})')

    dataset_test = prepare_train_data2(hist, symbol_out, istart_samples_test, n_samples_test, candle_input_type,
                                       candle_output_type)

    X, y = split_sequences2(dataset_test, n_steps, candle_output_type)

    model = load_model('model.h5')

    diffs = []
    len_X_ = len(X)
    for i in range(len_X_):
        print(f'{100 * i / len_X_:.2f} %')
        x_input = X[i]

        x_input = x_input.reshape((1, n_steps, n_features))

        # for MLP model only
        n_input = x_input.shape[1] * x_input.shape[2]
        x_input = x_input.reshape((x_input.shape[0], n_input))

        y_pred = model.predict(x_input)
        if len(y_pred[0]) == 1:
            diff = y[i] - y_pred[0][0]
        else:
            diff = y[i] - y_pred[0]
        diffs.append(diff)
        if i % 3000 == 0 and i > 0:
            _t = 60
            print(f'esperando {_t} segundos para continuar')
            time.sleep(_t)

    diffs = np.asarray(diffs)
    bias = np.mean(diffs, axis=0)
    print(f'bias = {bias}')

    train_config['bias'] = bias
    train_config['n_samples_test_for_calc_bias'] = n_samples_test
    write_train_config(train_config)
    print('bias salvo em train_config.json')


def test_model_with_trader():
    from TraderSimMultiNoPrints import TraderSimMulti

    settings = read_json('settings.json')
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    timeframe = settings['timeframe']
    hist = HistMulti(temp_dir, timeframe)

    train_config = read_train_config()
    print(f'train_config:')
    print(f'{train_config}')

    n_steps = train_config['n_steps']
    n_features = train_config['n_features']
    symbol_out = train_config['symbol_out']
    n_samples_train = train_config['n_samples_train']
    bias = train_config['bias']
    candle_input_type = train_config['candle_input_type']
    candle_output_type = train_config['candle_output_type']
    n_samples_test = 100
    samples_index_start = n_samples_train

    dataset_test = prepare_train_data(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type)
    X, y = split_sequences2(dataset_test, n_steps, candle_output_type)

    model = load_model('model.h5')

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    _symbol_timeframe = f'{symbol_out}_{hist.timeframe}'
    trans: MinMaxScaler = scalers[_symbol_timeframe]

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
        x_input = X[j]
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

    settings = read_json('settings.json')
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    timeframe = settings['timeframe']
    hist = HistMulti(temp_dir, timeframe)

    train_config = read_train_config()
    print(f'train_config:')
    print(f'{train_config}')

    n_steps = train_config['n_steps']
    n_features = train_config['n_features']
    symbol_out = train_config['symbol_out']
    n_samples_train = train_config['n_samples_train']
    bias = train_config['bias']
    candle_input_type = train_config['candle_input_type']
    candle_output_type = train_config['candle_output_type']
    n_samples_test = 100
    samples_index_start = n_samples_train

    dataset_test = prepare_train_data2(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type,
                                       candle_output_type)
    X, y = split_sequences2(dataset_test, n_steps, candle_output_type)

    model = load_model('model.h5')

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    _symbol_timeframe = f'{symbol_out}_{hist.timeframe}'
    trans: MinMaxScaler = scalers[_symbol_timeframe]

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
        x_input = X[j]
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
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))


if __name__ == '__main__':
    show_tf()
    train_model()
    # calculate_model_bias()
    # evaluate_model()
    # test_model_with_trader()
    # test_model_with_trader_interactive()
