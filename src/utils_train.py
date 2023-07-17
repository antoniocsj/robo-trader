import json
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
from HistMulti import HistMulti
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

tf.keras.utils.set_random_seed(1)

from utils_nn import split_sequences2, prepare_train_data2
from utils_filesystem import read_json
from utils_symbols import calc_n_inputs


def train_model_param(settings: dict, hist: HistMulti, n_steps: int, layer_type: list):
    csv_dir = settings['csv_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    candle_input_type = settings['candle_input_type']
    candle_output_type = settings['candle_output_type']

    n_steps = n_steps
    n_samples_train = 1000  # 30000-M10, 60000-M5
    validation_split = 0.2

    n_cols, n_symbols = calc_n_inputs(csv_dir, candle_input_type, timeframe)
    n_inputs = n_steps * n_cols
    n_epochs = 2

    print(f'n_steps = {n_steps}, candle_input_type = {candle_input_type}, candle_output_type = {candle_output_type}, '
          f'n_inputs = {n_inputs}, n_samples_train = {n_samples_train}, validation_split = {validation_split}, '
          f'n_epochs = {n_epochs}, layer_type = {layer_type}')

    # horizontally stack columns
    dataset_train = prepare_train_data2(hist, symbol_out, 0, n_samples_train, candle_input_type, candle_output_type)

    # convert into input/output
    X_train, y_train = split_sequences2(dataset_train, n_steps, candle_output_type)
    print(X_train.shape, y_train.shape)

    # We are now ready to fit a 1D CNN model on this data, specifying the expected number of time steps and
    # features to expect for each input sample.
    n_features = X_train.shape[2]

    if n_cols != n_features:
        print(f'ERRO. n_cols ({n_cols}) != n_features ({n_features}).')
        exit(-1)

    # define model
    model = Sequential()
    model.add(Conv1D(filters=n_features, kernel_size=n_steps, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Flatten())

    if sum(layer_type) > 0:
        for i in range(len(layer_type)):
            if layer_type[i] != 0:
                model.add(Dense(layer_type[i], activation='relu'))
    model.add(Dense(len(candle_output_type), activation='relu'))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=n_epochs, verbose=0, validation_split=validation_split)

    whole_set_train_loss_eval = model.evaluate(X_train, y_train, verbose=0)
    print(f'whole_set_train_loss_eval: {whole_set_train_loss_eval:} (n_samples_train = {n_samples_train})')

    n_samples_test = 100
    samples_index_start = n_samples_train
    dataset_test = prepare_train_data2(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type,
                                       candle_output_type)
    X_test, y_test = split_sequences2(dataset_test, n_steps, candle_output_type)

    test_loss_eval = model.evaluate(X_test, y_test, verbose=0)
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
                    'n_epochs': n_epochs,
                    'layer_type': layer_type,
                    'whole_set_train_loss_eval': whole_set_train_loss_eval,
                    'n_samples_test': n_samples_test,
                    'test_loss_eval': test_loss_eval,
                    'symbols': hist.symbols}

    return train_config


def test_models():
    settings = read_json('settings.json')
    print(f'settings.json: {settings}')

    csv_dir = settings['csv_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    candle_input_type = settings['candle_input_type']
    candle_output_type = settings['candle_output_type']

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
            train_configs = json.load(file)
            i_start = len(train_configs)
    else:
        train_configs = []
        i_start = 0

    wait = 1
    _len_layers_comb = len(layers_comb)
    for i in range(i_start, len(layers_comb)):
        _layer_type = sorted(list(layers_comb[i]), reverse=True)
        print(f'testando modelo com _layer_type = {_layer_type}, len_layers_comb = {_len_layers_comb}. '
              f'({100 * i / _len_layers_comb:.2f} %)')
        _out = train_model_param(settings, hist, n_steps, _layer_type)

        train_configs.append(_out)
        with open('test_models.json', 'w') as file:
            json.dump(train_configs, file, indent=4)

        print(f'esperando {wait} segundos')
        time.sleep(wait)
