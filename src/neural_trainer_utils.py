import os
import shutil
from time import time


def train_model(working_dir: str, settings: dict, params_rs_search: dict, seed: int, patience_style: str):
    """

    :param settings:
    :param params_rs_search:
    :param seed:
    :param patience_style: short or long
    :param working_dir:
    :return:
    """
    random_seed = seed

    os.environ['PYTHONHASHSEED'] = str(1)
    os.environ['TF_CUDNN_DETERMINISM'] = str(1)
    os.environ['TF_DETERMINISTIC_OPS'] = str(1)

    import numpy as np

    np.random.seed(random_seed)

    import random

    random.seed(random_seed)

    import tensorflow as tf
    import keras

    tf.random.set_seed(random_seed)
    keras.utils.set_random_seed(random_seed)

    from keras.api.models import Sequential, load_model
    from keras.api.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, LSTM
    from keras.api.callbacks import EarlyStopping, ModelCheckpoint
    from HistMulti import HistMulti
    from src.utils.utils_nn import split_sequences

    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))

    print(f'random_seed = {seed}')
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    candle_input_type = settings['candle_input_type']
    candle_output_type = settings['candle_output_type']
    csv_content = settings['csv_content']

    hist = HistMulti(temp_dir, timeframe, symbolout=symbol_out)
    datetime_start = hist.arr[symbol_out][timeframe][0][0]
    datetime_end = hist.arr[symbol_out][timeframe][-1][0]

    if hist.timeframe != timeframe:
        print(f'o timeframe do diretório {temp_dir} ({hist.timeframe}) é diferente do timeframe especificado '
              f'em settings.json ({timeframe})')
        exit(-1)

    # Create save directory if not exists
    results_dir = os.path.join(working_dir, settings['results_dir'])

    n_steps: int = params_rs_search['n_steps']
    n_hidden_layers: int = params_rs_search['n_hidden_layers']
    validation_split = settings['validation_split']
    samples_test_ratio = settings['samples_test_ratio']

    n_rows = hist.arr[symbol_out][timeframe].shape[0]

    n_samples_test = int(n_rows * samples_test_ratio)  # Número de amostras inéditas usadas na fase de avaliação.
    n_samples_train = n_rows - n_samples_test  # Número de amostras usadas na fase de treinamento e validação
    dataset_test, dataset_train = prepare_samples(hist, symbol_out, csv_content, candle_input_type, candle_output_type,
                                                  n_samples_train, n_samples_test)

    # convert into input/output samples
    X_train, y_train = split_sequences(dataset_train, n_steps, candle_output_type)
    X_test, y_test = split_sequences(dataset_test, n_steps, candle_output_type)

    # We are now ready to fit a 1D CNN model on this data, specifying the expected number of time steps and
    # features to expect for each input sample.
    n_features = X_train.shape[2]
    n_inputs = n_steps * n_features
    max_n_epochs = params_rs_search['max_n_epochs']

    if patience_style.lower() == 'short':
        patience = params_rs_search['patience_short']
    elif patience_style.lower() == 'long':
        patience = params_rs_search['patience_long']
    else:
        print(f'ERRO. patience_style ({patience_style}) inválido')
        exit(-1)

    n_symbols = len(hist.symbols)

    print(f'n_symbols = {n_symbols}')
    print(f'symbols = {hist.symbols}')
    print(f'tipo_vela_entrada = {candle_input_type}, n_steps = {n_steps}, n_hidden_layers = {n_hidden_layers}\n'
          f'tipo_vela_saída = {candle_output_type}, max_n_epochs = {max_n_epochs}, patience = {patience}\n'
          f'n_features (n_cols) = {n_features}, n_inputs = {n_inputs}\n'
          f'validation_split = {validation_split}, samples_test_ratio = {samples_test_ratio}\n'
          f'n_samples_train = {n_samples_train}, n_samples_test = {n_samples_test}')

    model_type = settings['model_type']

    # Instantiate model
    model = Sequential()
    n_neurons = n_inputs

    # input layer
    model.add(Input(shape=(n_steps, n_features)))

    if model_type == 'CNN':
        n_filters = n_features
        kernel_size = n_steps
        pool_size = n_inputs
        model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu'))
        # model.add(MaxPooling1D(pool_size=pool_size, padding='same'))
        model.add(AveragePooling1D(pool_size=pool_size, padding='same'))
        model.add(Flatten())
    elif model_type == 'LSTM':
        model.add(LSTM(n_inputs, activation='relu'))
    else:
        print(f'ERRO. model_type ({model_type}) inválido.')
        exit(-1)

    # hidden layers
    for i in range(n_hidden_layers):
        model.add(Dense(n_neurons, activation='relu'))

    # output layer
    model.add(Dense(len(candle_output_type)))
    model.compile(optimizer='adam', loss='mse')
    model_config = model.get_config()

    print(f'model_type = {model_type}.')

    # fit model
    print(f'treinando o modelo em parte das amostras de treinamento.')
    print(f'n_samples_train * validation_split = {n_samples_train} * {validation_split} = '
          f'{int(n_samples_train * validation_split)}).')

    model_filepath = os.path.join(results_dir, 'model.keras')

    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
                 ModelCheckpoint(filepath=model_filepath, monitor='val_loss', save_best_only=True, verbose=1)]

    # Fit model
    t0 = time()
    history = model.fit(X_train, y_train, epochs=max_n_epochs, verbose=1, validation_split=validation_split, callbacks=callbacks)
    training_time = time() - t0
    print('Training time: ', training_time)

    effective_n_epochs = len(history.history['loss'])
    loss, val_loss = history.history['loss'], history.history['val_loss']
    i_min_loss, i_min_val_loss = np.argmin(loss), np.argmin(val_loss)
    min_loss, min_val_loss = loss[i_min_loss], val_loss[i_min_val_loss]
    losses = {'min_loss': {'value': min_loss, 'index': i_min_loss, 'epoch': i_min_loss + 1},
              'min_val_loss': {'value': min_val_loss, 'index': i_min_val_loss, 'epoch': i_min_val_loss + 1}}

    print(f'avaliando o modelo no conjunto inteiro das amostras de treinamento.')
    saved_model = load_model(model_filepath)
    whole_set_train_loss_eval = saved_model.evaluate(X_train, y_train, verbose=0)
    print(f'whole_set_train_loss_eval: {whole_set_train_loss_eval:} (n_samples_train = {n_samples_train})')

    print(f'avaliando o modelo num novo conjunto de amostras de teste.')
    saved_model = load_model(model_filepath)
    test_loss_eval = saved_model.evaluate(X_test, y_test, verbose=0)
    print(f'test_loss_eval: {test_loss_eval} (n_samples_test = {n_samples_test})')

    product = whole_set_train_loss_eval * test_loss_eval
    print(f'p_{random_seed} = {whole_set_train_loss_eval} * {test_loss_eval} = {product} patience={patience} '
          f'eff_n_epochs={effective_n_epochs}')

    train_config = {
        'symbol_out': symbol_out,
        'timeframe': hist.timeframe,
        'n_steps': n_steps,
        'candle_input_type': candle_input_type,
        'candle_output_type': candle_output_type,
        'n_symbols': n_symbols,
        'n_features': n_features,
        'n_inputs': n_inputs,
        'n_hidden_layers': n_hidden_layers,
        'random_seed': random_seed,
        'n_samples_train': n_samples_train,
        'validation_split': validation_split,
        'effective_n_epochs': effective_n_epochs,
        'max_n_epochs': max_n_epochs,
        'patience': patience,
        'datetime_start': datetime_start,
        'datetime_end': datetime_end,
        'training_time': training_time,
        'whole_set_train_loss_eval': whole_set_train_loss_eval,
        'samples_test_ratio': samples_test_ratio,
        'n_samples_test': n_samples_test,
        'test_loss_eval': test_loss_eval,
        'losses': losses,
        'symbols': hist.symbols,
        'bias': 0.0,
        'n_samples_test_for_calc_bias': 0,
        'model_config': model_config,
        'history': history.history
    }

    return train_config


def prepare_samples(hist, symbol_out, csv_content, candle_input_type, candle_output_type, n_samples_train, n_samples_test):
    """
    :param hist: The historical data used to prepare the samples.
    :param symbol_out: The symbol or identifier of the output.
    :param csv_content: The type of CSV content. It can be either 'HETEROGENEOUS_OHLCV' or 'HETEROGENEOUS_DEFAULT'.
    :param candle_input_type: The type of input candle.
    :param candle_output_type: The type of output candle.
    :param n_samples_train: The number of samples for training.
    :param n_samples_test: The number of samples for testing.
    :return: A tuple containing the training and testing datasets.

    """
    from src.utils.utils_nn import prepare_train_data_candles, prepare_train_data_indicators

    samples_index_start = n_samples_train - 1

    if csv_content == 'HETEROGENEOUS_OHLCV':
        dataset_train = prepare_train_data_candles(hist, symbol_out, 0, n_samples_train, candle_input_type, candle_output_type)
        dataset_test = prepare_train_data_candles(hist, symbol_out, samples_index_start, n_samples_test, candle_input_type,
                                                  candle_output_type)
    elif csv_content == 'HETEROGENEOUS_DEFAULT':
        dataset_train = prepare_train_data_indicators(hist, symbol_out, 0, n_samples_train)
        dataset_test = prepare_train_data_indicators(hist, symbol_out, samples_index_start, n_samples_test)
    else:
        print(f'ERRO. csv_content ({csv_content}) inválido.')
        exit(-1)

    return dataset_train, dataset_test


def get_time_break_from_timeframe(tf: str):
    """
    Define uma tabela que relaciona o tempo de intervalo entre os treinos das redes neurais em função do timeframe
    dos dados históricos. O objetivo dessa pausa entre os treinos serve para não superaquecer a placa de vídeo.
    # supondo 5 anos de histórico
    # TF(MIN)    PAUSA(S)
    #  5        80
    # 10        70
    # 15        60
    # 20        50
    # 30        40
    # 60        30
    :return:
    """
    factor = 2

    if tf == 'M5':
        ret = 360
    elif tf == 'M10':
        ret = 180
    elif tf == 'M15':
        ret = 120
    elif tf == 'M20':
        ret = 90
    elif tf == 'M30':
        ret = 60
    elif tf == 'H1':
        ret = 30
    else:
        print('ERRO. get_time_break_from_timeframe. timeframe inválido.')
        exit(-1)

    return ret * factor
