import numpy as np
from numpy import ndarray
from HistMulti import HistMulti


# usada nas criação de amostra de treinamento das redes neurais
def prepare_train_data_0(_hist: HistMulti, _symbol_out: str, _start_index: int,
                         _num_candles: int, _candle_type: str) -> ndarray:
    _data = []
    _timeframe = _hist.timeframe
    _symbol_tf_out = f'{_symbol_out}_{_timeframe}'

    if _candle_type == 'C':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            _c_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_candles][:, 4]
            _c_in = _c_in.reshape(len(_c_in), 1)
            if len(_data) == 0:
                _data = _c_in
            else:
                _data = np.hstack((_data, _c_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_candles + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _candle_type == 'CV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            _cv_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_candles][:, 4:6]
            if len(_data) == 0:
                _data = _cv_in
            else:
                _data = np.hstack((_data, _cv_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_candles + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _candle_type == 'OHLC':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            _ohlc_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_candles][:, 1:5]
            if len(_data) == 0:
                _data = _ohlc_in
            else:
                _data = np.hstack((_data, _ohlc_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_candles + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _candle_type == 'OHLCV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            _ohlcv_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_candles][:, 1:6]
            if len(_data) == 0:
                _data = _ohlcv_in
            else:
                _data = np.hstack((_data, _ohlcv_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_candles + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    else:
        print(f'tipo de vela de entrada não suportado: {_candle_type}')
        exit(-1)

    return _data


# usada nas criação de amostra de treinamento das redes neurais
def prepare_train_data(_hist: HistMulti, _symbol_out: str, _start_index: int,
                       _num_candles: int, _candle_input_type: str) -> ndarray:
    _data = []
    _timeframe = _hist.timeframe
    _symbol_tf_out = f'{_symbol_out}_{_timeframe}'

    if _candle_input_type == 'C':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 2:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_candles][:, 1]
            else:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_candles][:, 4]
            _data_in = _data_in.reshape(len(_data_in), 1)
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_candles + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _candle_input_type == 'CV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 2:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_candles][:, 1]
                _data_in = _data_in.reshape(len(_data_in), 1)
            else:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_candles][:, 4:6]
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_candles + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _candle_input_type == 'OHLC':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 2:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_candles][:, 1]
                _data_in = _data_in.reshape(len(_data_in), 1)
            else:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_candles][:, 1:5]
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_candles + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _candle_input_type == 'OHLCV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 2:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_candles][:, 1]
                _data_in = _data_in.reshape(len(_data_in), 1)
            else:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_candles][:, 1:6]
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_candles + 1][:, 4]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    else:
        print(f'tipo de vela de entrada não suportado: {_candle_input_type}')
        exit(-1)

    return _data


# usada nas criação de amostra de treinamento das redes neurais
def prepare_train_data2(hist: HistMulti, symbol_out: str, start_index: int, num_candles: int,
                        candle_input_type: str, candle_output_type: str) -> ndarray:
    _data = []
    timeframe = hist.timeframe
    symbol_tf_out = f'{symbol_out}_{timeframe}'

    if candle_input_type == 'C':
        for symbol in hist.symbols:
            _symbol_timeframe = f'{symbol}_{timeframe}'
            if hist.arr[_symbol_timeframe].shape[1] == 2:
                data_in = hist.arr[_symbol_timeframe][start_index:start_index + num_candles][:, 1]
            else:
                data_in = hist.arr[_symbol_timeframe][start_index:start_index + num_candles][:, 4]
            data_in = data_in.reshape(len(data_in), 1)
            if len(_data) == 0:
                _data = data_in
            else:
                _data = np.hstack((_data, data_in))

        if candle_output_type == 'C':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 4]
            data_out = data_out.reshape(len(data_out), 1)
            _data = np.hstack((_data, data_out))
        else:
            print(f'ERRO. candle_output_type ({candle_output_type}) não suportado para candle_input_type = '
                  f'{candle_input_type}')
            exit(-1)

    elif candle_input_type == 'CV':
        for symbol in hist.symbols:
            _symbol_timeframe = f'{symbol}_{timeframe}'
            if hist.arr[_symbol_timeframe].shape[1] == 2:
                data_in = hist.arr[_symbol_timeframe][start_index:start_index + num_candles][:, 1]
                data_in = data_in.reshape(len(data_in), 1)
            else:
                data_in = hist.arr[_symbol_timeframe][start_index:start_index + num_candles][:, 4:6]
            if len(_data) == 0:
                _data = data_in
            else:
                _data = np.hstack((_data, data_in))

        if candle_output_type == 'C':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 4]
            data_out = data_out.reshape(len(data_out), 1)
        elif candle_output_type == 'CV':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 4:6]
        else:
            print(f'ERRO. candle_output_type ({candle_output_type}) não suportado para candle_input_type = '
                  f'{candle_input_type}')
            exit(-1)

        _data = np.hstack((_data, data_out))

    elif candle_input_type == 'HLC':
        for symbol in hist.symbols:
            _symbol_timeframe = f'{symbol}_{timeframe}'
            if hist.arr[_symbol_timeframe].shape[1] == 2:
                data_in = hist.arr[_symbol_timeframe][start_index:start_index + num_candles][:, 1]
                data_in = data_in.reshape(len(data_in), 1)
            else:
                data_in = hist.arr[_symbol_timeframe][start_index:start_index + num_candles][:, 2:5]
            if len(_data) == 0:
                _data = data_in
            else:
                _data = np.hstack((_data, data_in))

        if candle_output_type == 'C':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 4]
            data_out = data_out.reshape(len(data_out), 1)
        elif candle_output_type == 'HL':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:4]
        elif candle_output_type == 'HLC':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:5]
        else:
            print(f'ERRO. candle_output_type ({candle_output_type}) não suportado para candle_input_type = '
                  f'{candle_input_type}')
            exit(-1)

        _data = np.hstack((_data, data_out))

    elif candle_input_type == 'HLCV':
        for symbol in hist.symbols:
            _symbol_timeframe = f'{symbol}_{timeframe}'
            if hist.arr[_symbol_timeframe].shape[1] == 2:
                data_in = hist.arr[_symbol_timeframe][start_index:start_index + num_candles][:, 1]
                data_in = data_in.reshape(len(data_in), 1)
            else:
                data_in = hist.arr[_symbol_timeframe][start_index:start_index + num_candles][:, 2:6]
            if len(_data) == 0:
                _data = data_in
            else:
                _data = np.hstack((_data, data_in))

        if candle_output_type == 'C':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 4]
            data_out = data_out.reshape(len(data_out), 1)
        elif candle_output_type == 'CV':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 4:6]
        elif candle_output_type == 'HL':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:4]
        elif candle_output_type == 'HLV':
            # data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:4 e 5]
            a = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:4]
            b = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 5]
            b = b.reshape(len(b), 1)
            data_out = np.hstack((a, b))
        elif candle_output_type == 'HLC':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:5]
        elif candle_output_type == 'HLCV':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:6]
        else:
            print(f'ERRO. candle_output_type ({candle_output_type}) não suportado para candle_input_type = '
                  f'{candle_input_type}')
            exit(-1)

        _data = np.hstack((_data, data_out))

    elif candle_input_type == 'OHLC':
        for symbol in hist.symbols:
            _symbol_timeframe = f'{symbol}_{timeframe}'
            if hist.arr[_symbol_timeframe].shape[1] == 2:
                data_in = hist.arr[_symbol_timeframe][start_index:start_index + num_candles][:, 1]
                data_in = data_in.reshape(len(data_in), 1)
            else:
                data_in = hist.arr[_symbol_timeframe][start_index:start_index + num_candles][:, 1:5]
            if len(_data) == 0:
                _data = data_in
            else:
                _data = np.hstack((_data, data_in))

        if candle_output_type == 'C':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 4]
            data_out = data_out.reshape(len(data_out), 1)
        elif candle_output_type == 'HL':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:4]
        elif candle_output_type == 'HLC':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:5]
        elif candle_output_type == 'OHLC':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 1:5]
        else:
            print(f'ERRO. candle_output_type ({candle_output_type}) não suportado para candle_input_type = '
                  f'{candle_input_type}')
            exit(-1)

        _data = np.hstack((_data, data_out))

    elif candle_input_type == 'OHLCV':
        for symbol in hist.symbols:
            _symbol_timeframe = f'{symbol}_{timeframe}'
            if hist.arr[_symbol_timeframe].shape[1] == 2:
                data_in = hist.arr[_symbol_timeframe][start_index:start_index + num_candles][:, 1]
                data_in = data_in.reshape(len(data_in), 1)
            else:
                data_in = hist.arr[_symbol_timeframe][start_index:start_index + num_candles][:, 1:6]
            if len(_data) == 0:
                _data = data_in
            else:
                _data = np.hstack((_data, data_in))

        if candle_output_type == 'C':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 4]
            data_out = data_out.reshape(len(data_out), 1)
        elif candle_output_type == 'CV':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 4:6]
        elif candle_output_type == 'HL':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:4]
        elif candle_output_type == 'HLV':
            # data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:4 e 5]
            a = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:4]
            b = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 5]
            b = b.reshape(len(b), 1)
            data_out = np.hstack((a, b))
        elif candle_output_type == 'HLC':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:5]
        elif candle_output_type == 'HLCV':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 2:6]
        elif candle_output_type == 'OHLCV':
            data_out = hist.arr[symbol_tf_out][start_index + 1:start_index + num_candles + 1][:, 1:6]
        else:
            print(f'ERRO. candle_output_type ({candle_output_type}) não suportado para candle_input_type = '
                  f'{candle_input_type}')
            exit(-1)

        _data = np.hstack((_data, data_out))

    else:
        print(f'tipo de vela de entrada não suportado: {candle_input_type}')
        exit(-1)

    return _data


# Multiple Input Series
# split a multivariate sequence into input/output samples.
# We can define a function named split_sequences1() that will take a dataset as we have defined it with rows for
# time steps and columns for parallel series and return input/output samples.
def split_sequences1(sequences: ndarray, n_steps: int) -> tuple[ndarray, ndarray]:
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


# Multiple Parallel Series
# split a multivariate sequence into input/output samples.
def split_sequences2(sequences: ndarray, n_steps: int, candle_output_type: str) -> tuple[ndarray, ndarray]:
    out_len = len(candle_output_type)
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break

        # cada linha da tabela sequences contém uma amostra entrada/saída.
        # os elementos que formam a entrada ficam nas primeiras posições da linha.
        # os elementos que formam a saída ficam nas últimas posições da linha.
        if out_len == 1:
            # nesse caso, a saída é um escalar (dimensão=1), então pegue apenas o último elemento do array.
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        else:
            # nesse caso, a saída é um array (dimensão=out_len), então pegue os out_len últimos elemento do array.
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-out_len], sequences[end_ix - 1, -out_len:]

        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# usada nas preparação dos dados históricos de entrada da rede neural para previsão
def prepare_data_for_prediction(_hist: HistMulti, _num_candles: int, _candle_type: str) -> ndarray:
    _data = []
    _timeframe = _hist.timeframe

    if _candle_type == 'C':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 2:
                _data_in = _hist.arr[_symbol_timeframe][0:_num_candles][:, 1]
            else:
                _data_in = _hist.arr[_symbol_timeframe][0:_num_candles][:, 4]
            _data_in = _data_in.reshape(len(_data_in), 1)
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))
    elif _candle_type == 'CV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 2:
                _data_in = _hist.arr[_symbol_timeframe][0:_num_candles][:, 1]
                _data_in = _data_in.reshape(len(_data_in), 1)
            else:
                _data_in = _hist.arr[_symbol_timeframe][0:_num_candles][:, 4:6]
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))
    elif _candle_type == 'OHLC':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 2:
                _data_in = _hist.arr[_symbol_timeframe][0:_num_candles][:, 1]
                _data_in = _data_in.reshape(len(_data_in), 1)
            else:
                _data_in = _hist.arr[_symbol_timeframe][0:_num_candles][:, 1:5]
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))
    elif _candle_type == 'OHLCV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 2:
                _data_in = _hist.arr[_symbol_timeframe][0:_num_candles][:, 1]
                _data_in = _data_in.reshape(len(_data_in), 1)
            else:
                _data_in = _hist.arr[_symbol_timeframe][0:_num_candles][:, 1:6]
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))
    else:
        print(f'tipo de vela não suportado: {_candle_type}')
        exit(-1)

    return _data
