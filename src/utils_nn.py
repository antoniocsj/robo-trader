import numpy as np
from numpy import ndarray
from HistMulti import HistMulti


# usada nas criação de amostra de treinamento das redes neurais
def prepare_train_data_multi_0(_hist: HistMulti, _symbol_out: str, _start_index: int,
                               _num_velas: int, _tipo_vela: str) -> ndarray:
    _data = []
    _timeframe = _hist.timeframe
    _symbol_tf_out = f'{_symbol_out}_{_timeframe}'

    if _tipo_vela == 'C':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            _c_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 4]
            _c_in = _c_in.reshape(len(_c_in), 1)
            if len(_data) == 0:
                _data = _c_in
            else:
                _data = np.hstack((_data, _c_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _tipo_vela == 'CV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            _cv_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 4:6]
            if len(_data) == 0:
                _data = _cv_in
            else:
                _data = np.hstack((_data, _cv_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _tipo_vela == 'OHLC':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            _ohlc_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 1:5]
            if len(_data) == 0:
                _data = _ohlc_in
            else:
                _data = np.hstack((_data, _ohlc_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _tipo_vela == 'OHLCV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            _ohlcv_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 1:6]
            if len(_data) == 0:
                _data = _ohlcv_in
            else:
                _data = np.hstack((_data, _ohlcv_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    else:
        print(f'tipo de vela não suportado: {_tipo_vela}')
        exit(-1)

    return _data


# usada nas criação de amostra de treinamento das redes neurais
def prepare_train_data_multi(_hist: HistMulti, _symbol_out: str, _start_index: int,
                             _num_velas: int, _tipo_vela: str) -> ndarray:
    _data = []
    _timeframe = _hist.timeframe
    _symbol_tf_out = f'{_symbol_out}_{_timeframe}'

    if _tipo_vela == 'C':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 2:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 1]
            else:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 4]
            _data_in = _data_in.reshape(len(_data_in), 1)
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _tipo_vela == 'CV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 2:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 1]
                _data_in = _data_in.reshape(len(_data_in), 1)
            else:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 4:6]
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _tipo_vela == 'OHLC':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 2:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 1]
                _data_in = _data_in.reshape(len(_data_in), 1)
            else:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 1:5]
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _tipo_vela == 'OHLCV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 2:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 1]
                _data_in = _data_in.reshape(len(_data_in), 1)
            else:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 1:6]
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 4]
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
