import os
import time

from utils_filesystem import read_json, write_json


def train_model(deterministic: bool = True, seed: int = 1):
    deterministic = deterministic
    random_seed = seed

    # import os

    if deterministic:
        os.environ['PYTHONHASHSEED'] = str(1)
        os.environ['TF_CUDNN_DETERMINISM'] = str(1)
        os.environ['TF_DETERMINISTIC_OPS'] = str(1)

    import numpy as np

    if deterministic:
        np.random.seed(random_seed)

    import random

    if deterministic:
        random.seed(random_seed)

    import tensorflow as tf

    if deterministic:
        tf.random.set_seed(random_seed)
        tf.keras.utils.set_random_seed(random_seed)

    from HistMulti import HistMulti
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers import Conv1D
    from keras.layers import MaxPooling1D
    from keras.models import load_model
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    if deterministic:
        tf.keras.utils.set_random_seed(random_seed)

    from utils_nn import split_sequences2, prepare_train_data2

    print(os.environ["LD_LIBRARY_PATH"])
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))

    print(f'deterministic? = {deterministic}')
    if deterministic:
        print(f'random_seed = {seed}')

    settings = read_json('settings.json')
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    candle_input_type = settings['candle_input_type']
    candle_output_type = settings['candle_output_type']
    hist = HistMulti(temp_dir, timeframe, symbols_allowed=[symbol_out])

    if hist.timeframe != timeframe:
        print(f'o timeframe do diretório {temp_dir} ({hist.timeframe}) é diferente do timeframe especificado '
              f'em settings.json ({timeframe})')
        exit(-1)

    n_steps = 2
    n_hidden_layers = 1

    samples_test_ratio = 0.02

    n_rows = hist.arr[symbol_out][timeframe].shape[0]

    # Número de amostras inéditas usadas na fase de avaliação.
    n_samples_test = int(n_rows * samples_test_ratio)

    n_samples_train = n_rows - n_samples_test  # Número de amostras usadas na fase de treinamento e validação
    validation_split = 0.2

    # horizontally stack columns
    dataset_train = prepare_train_data2(hist, symbol_out, 0, n_samples_train, candle_input_type, candle_output_type)

    # convert into input/output samples
    X_train, y_train = split_sequences2(dataset_train, n_steps, candle_output_type)

    # We are now ready to fit a 1D CNN model on this data, specifying the expected number of time steps and
    # features to expect for each input sample.
    n_features = X_train.shape[2]
    n_inputs = n_steps * n_features
    max_n_epochs = n_inputs * 3 * 0 + 150
    patience = int(max_n_epochs / 10) * 0 + 5
    n_symbols = len(hist.symbols)

    print(f'n_symbols = {n_symbols}')
    print(f'symbols = {hist.symbols}')
    print(f'tipo_vela_entrada = {candle_input_type}, n_steps = {n_steps}, n_hidden_layers = {n_hidden_layers}\n'
          f'tipo_vela_saída = {candle_output_type}, max_n_epochs = {max_n_epochs}, patience = {patience}\n'
          f'n_features (n_cols) = {n_features}, n_inputs = {n_inputs}\n'
          f'validation_split = {validation_split}, samples_test_ratio = {samples_test_ratio}\n'
          f'n_samples_train = {n_samples_train}, n_samples_test = {n_samples_test}')

    model = Sequential()
    n_filters = n_features
    kernel_size = n_steps
    pool_size = n_inputs
    n_neurons = n_inputs

    # define cnn model
    # input layer
    model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=pool_size, padding='same'))
    model.add(Flatten())

    # hidden layers
    for i in range(n_hidden_layers):
        model.add(Dense(n_neurons, activation='relu'))

    # define MLP model
    # n_input = X_train.shape[1] * X_train.shape[2]
    # X_train = X_train.reshape((X_train.shape[0], n_input))
    # model.add(Dense(n_inputs, activation='relu', input_dim=n_input))
    # model.add(Dense(n_inputs, activation='relu'))

    # output layer
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
    samples_index_start = n_samples_train - 1
    dataset_test = prepare_train_data2(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type,
                                       candle_output_type)

    X_test, y_test = split_sequences2(dataset_test, n_steps, candle_output_type)

    # for MLP model only
    # n_input = X_test.shape[1] * X_test.shape[2]
    # X_test = X_test.reshape((X_test.shape[0], n_input))

    saved_model = load_model('model.h5')
    test_loss_eval = saved_model.evaluate(X_test, y_test, verbose=0)
    print(f'test_loss_eval: {test_loss_eval} (n_samples_test = {n_samples_test})')

    product = whole_set_train_loss_eval * test_loss_eval
    print(f'p_{random_seed} = {whole_set_train_loss_eval} * {test_loss_eval} = {product} patience={patience}')

    train_config = {'symbol_out': symbol_out,
                    'timeframe': hist.timeframe,
                    'n_steps': n_steps,
                    'candle_input_type': candle_input_type,
                    'candle_output_type': candle_output_type,
                    'n_symbols': n_symbols,
                    'n_features': n_features,
                    'n_inputs': n_inputs,
                    'n_hidden_layers': n_hidden_layers,
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

    return train_config


def create_train_log():
    import os
    _filename = 'train_log.json'
    if not os.path.exists(_filename):
        print(f'o arquivo {_filename} não existe ainda. será criado agora.')
        _dict = {
            'n_experiments': 0,
            'experiments': []
        }
        write_json(_filename, _dict)
    else:
        print(f'o arquivo {_filename} já existe. continuando os experimentos.')


def load_train_log() -> dict:
    _filename = 'train_log.json'
    if os.path.exists(_filename):
        _dict = read_json(_filename)
        return _dict
    else:
        print(f'ERRO. o arquivo {_filename} não existe.')
        exit(-1)


def update_train_log(train_log: dict):
    _filename = 'train_log.json'
    if os.path.exists(_filename):
        write_json(_filename, train_log)
    else:
        print(f'ERRO. o arquivo {_filename} não existe.')
        exit(-1)


def trainer_01():
    create_train_log()
    _secs = 80

    while True:
        train_log = load_train_log()
        index = train_log['n_experiments'] + 1

        train_config = train_model(seed=index)

        whole_set_train_loss_eval = train_config['whole_set_train_loss_eval']
        test_loss_eval = train_config['test_loss_eval']
        product = whole_set_train_loss_eval * test_loss_eval

        log = {
            'random_seed': index,
            'whole_set_train_loss': whole_set_train_loss_eval,
            'test_loss': test_loss_eval,
            'product': product
        }
        train_log['n_experiments'] = index
        train_log['experiments'].append(log)
        update_train_log(train_log=train_log)

        print(f'esperando por {_secs} segundos')
        time.sleep(_secs)


if __name__ == '__main__':
    trainer_01()
