from typing import Any

from numpy import ndarray
import numpy as np
import pandas as pd
import pickle
from HistMulti import HistMulti
from sklearn.preprocessing import MinMaxScaler
from utils_filesystem import read_json


def denorm_close_price(_c, scaler: MinMaxScaler):
    c_denorm = scaler.inverse_transform(np.array([0, 0, 0, _c, 0], dtype=object).reshape(1, -1))
    c_denorm = c_denorm[0][3]
    return c_denorm


def denorm_output_array(_a: ndarray, scaler: MinMaxScaler) -> ndarray:
    output = scaler.inverse_transform(_a)
    output = output[0]
    return output


def denorm_output_(_out: Any, candle_type: str, scaler: MinMaxScaler) -> Any:
    if isinstance(_out, ndarray):
        if candle_type == 'OHLCV':
            output = scaler.inverse_transform(_out)
            output = output[0]
        elif candle_type == 'OHLC':
            _O, _H, _L, _C = _out[0], _out[1], _out[2], _out[3]
            output = scaler.inverse_transform(np.array([_O, _H, _L, _C, 0], dtype=object).reshape(1, -1))
            output = output[0]
        elif candle_type == 'HLCV':
            _H, _L, _C, _V = _out[0], _out[1], _out[2], _out[3]
            output = scaler.inverse_transform(np.array([0, _H, _L, _C, _V], dtype=object).reshape(1, -1))
            output = output[0]
        elif candle_type == 'HLC':
            _H, _L, _C = _out[0], _out[1], _out[2]
            output = scaler.inverse_transform(np.array([0, _H, _L, _C, 0], dtype=object).reshape(1, -1))
            output = output[0]
        elif candle_type == 'HLV':
            _H, _L, _V = _out[0], _out[1], _out[2]
            output = scaler.inverse_transform(np.array([0, _H, _L, 0, _V], dtype=object).reshape(1, -1))
            output = output[0]
        elif candle_type == 'HL':
            _H, _L = _out[0], _out[1]
            output = scaler.inverse_transform(np.array([0, _H, _L, 0, 0], dtype=object).reshape(1, -1))
            output = output[0]
        elif candle_type == 'CV':
            _C, _V = _out[0], _out[1]
            output = scaler.inverse_transform(np.array([0, 0, 0, _C, _V], dtype=object).reshape(1, -1))
            output = output[0]
        else:
            print(f'ERRO. denorm_output(). tipo de vela não suportado ({candle_type}).')
            exit(-1)
    else:
        if candle_type == 'C':
            _C = _out
            output = scaler.inverse_transform(np.array([0, 0, 0, _C, 0], dtype=object).reshape(1, -1))
            output = output[0][3]
        else:
            print(f'ERRO. denorm_output(). tipo de vela não suportado ({candle_type}).')
            exit(-1)

    return output


def denorm_output(arr: ndarray, bias: Any, candle_type: str, scaler: MinMaxScaler) -> Any:
    arr = arr[0] + bias

    if candle_type == 'OHLCV':
        _O, _H, _L, _C, _V = arr[0], arr[1], arr[2], arr[3], arr[4]
        output = scaler.inverse_transform(np.array([_O, _H, _L, _C, _V], dtype=object).reshape(1, -1))
        output = output[0]
    elif candle_type == 'OHLC':
        _O, _H, _L, _C = arr[0], arr[1], arr[2], arr[3]
        output = scaler.inverse_transform(np.array([_O, _H, _L, _C, 0], dtype=object).reshape(1, -1))
        output = output[0]
    elif candle_type == 'HLCV':
        _H, _L, _C, _V = arr[0], arr[1], arr[2], arr[3]
        output = scaler.inverse_transform(np.array([0, _H, _L, _C, _V], dtype=object).reshape(1, -1))
        output = output[0]
    elif candle_type == 'HLC':
        _H, _L, _C = arr[0], arr[1], arr[2]
        output = scaler.inverse_transform(np.array([0, _H, _L, _C, 0], dtype=object).reshape(1, -1))
        output = output[0]
    elif candle_type == 'HLV':
        _H, _L, _V = arr[0], arr[1], arr[2]
        output = scaler.inverse_transform(np.array([0, _H, _L, 0, _V], dtype=object).reshape(1, -1))
        output = output[0]
    elif candle_type == 'HL':
        _H, _L = arr[0], arr[1]
        output = scaler.inverse_transform(np.array([0, _H, _L, 0, 0], dtype=object).reshape(1, -1))
        output = output[0]
    elif candle_type == 'CV':
        _C, _V = arr[0], arr[1]
        output = scaler.inverse_transform(np.array([0, 0, 0, _C, _V], dtype=object).reshape(1, -1))
        output = output[0]
    elif candle_type == 'C':
        _C = arr[0]
        output = scaler.inverse_transform(np.array([0, 0, 0, _C, 0], dtype=object).reshape(1, -1))
        output = output[0][3]
    else:
        print(f'ERRO. denorm_output(). tipo de vela não suportado ({candle_type}).')
        exit(-1)

    return output


def normalize_directory(directory: str):
    setup = read_json('settings.json')
    timeframe = setup['timeframe']

    hist = HistMulti(directory, timeframe)
    scalers = {}

    for _symbol in hist.symbols:
        # print(_symbol)
        _symbol_timeframe = f'{_symbol}_{hist.timeframe}'

        arr: ndarray = hist.arr[_symbol_timeframe]
        if arr.shape[1] == 2:
            data: ndarray = arr[:, 1]
            data = data.reshape(len(data), 1)
        else:
            data = arr[:, 1:6]
        
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[:, 0], True)
        dataf.columns = range(dataf.columns.size)
        _filepath = hist.get_csv_filepath(_symbol_timeframe)
        dataf.to_csv(_filepath, index=False, sep='\t')
        scalers[_symbol_timeframe] = scaler

    with open('scalers.pkl', 'wb') as file:
        pickle.dump(scalers, file)

    print(f'todos os símbolos do diretório {directory} foram normalizados.')
    print('o arquivo scalers.pkl, que guarda as informações da normalização, foi salvo.')


def normalize_symbols(hist: HistMulti, scalers: dict, symbols: list[str] = None):
    for symbol in hist.symbols:
        if symbols and symbol not in symbols:
            continue

        # print(symbol)
        _symbol_timeframe = f'{symbol}_{hist.timeframe}'

        arr: ndarray = hist.arr[_symbol_timeframe]
        if arr.shape[1] == 2:
            data: ndarray = arr[:, 1]
            data = data.reshape(len(data), 1)
        else:
            data = arr[:, 1:6]

        scaler = scalers[_symbol_timeframe]
        data = scaler.transform(data)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[:, 0], True)
        dataf.columns = range(dataf.columns.size)
        hist.arr[_symbol_timeframe] = dataf.to_numpy(copy=True)

    if symbols:
        print(f'os símbolos {symbols} de {type(hist)} foram normalizados.')
        hist.update_sheets(symbols)
    else:
        hist.update_sheets()
        print(f'todos os símbolos de {type(hist)} foram normalizados.')


def differentiate_directory(directory: str):
    print(f'diferenciando diretório {directory}')

    setup = read_json('settings.json')
    timeframe = setup['timeframe']

    hist = HistMulti(directory, timeframe)

    for _symbol in hist.symbols:
        # print(_symbol)
        _symbol_timeframe = f'{_symbol}_{hist.timeframe}'

        arr = hist.arr[_symbol_timeframe]
        if arr.shape[1] == 2:
            data = arr[:, 1]
        else:
            data = arr[:, 1:6]

        data = np.diff(data, axis=0)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[1:, 0], True)
        dataf.columns = range(dataf.columns.size)
        _filepath = hist.get_csv_filepath(_symbol_timeframe)
        dataf.to_csv(_filepath, index=False, sep='\t')

    print(f'todos os símbolos do diretório {directory} foram diferenciados.')


def differentiate_files(filepath_list: list[str], directory: str):
    print(f'diferenciando alguns símbolos do diretório {directory}')

    for _filepath in filepath_list:
        # print(_filepath)
        df: pd.DataFrame = pd.read_csv(_filepath, sep='\t')
        arr = df.to_numpy()

        if arr.shape[1] == 2:
            data = arr[:, 1]
        else:
            data = arr[:, 1:6]

        data = np.diff(data, axis=0)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[1:, 0], True)
        dataf.columns = range(dataf.columns.size)
        dataf.to_csv(_filepath, index=False, sep='\t')

    print(f'{len(filepath_list)} símbolos do diretório {directory} foram diferenciados: {filepath_list}')


def differentiate_symbols(hist: HistMulti, symbols: list[str] = None):
    print(f'diferenciando alguns símbolos do objeto hist {type(hist)}')

    for symbol in hist.symbols:
        if symbols and symbol not in symbols:
            continue

        # print(symbol)
        _symbol_timeframe = f'{symbol}_{hist.timeframe}'

        arr = hist.arr[_symbol_timeframe]
        if arr.shape[1] == 2:
            data: ndarray = arr[:, 1]
            data = data.reshape(len(data), 1)
        else:
            data = arr[:, 1:6]

        data = np.diff(data, axis=0)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[1:, 0], True)
        dataf.columns = range(dataf.columns.size)
        hist.arr[_symbol_timeframe] = dataf.to_numpy(copy=True)

    if symbols:
        print(f'os símbolos {symbols} de {type(hist)} foram diferenciados.')
        hist.update_sheets(symbols)
    else:
        hist.update_sheets()
        print(f'todos os símbolos de {type(hist)} foram diferenciados.')


def apply_transform_str(arr: ndarray, transform_str: str) -> ndarray:
    if arr.shape[1] != 6:
        print('ERRO. apply_transform_str(). arr.shape[2] != 6.')
        exit(-1)

    if transform_str == '(C-O)*V':
        out_arr = (arr[:, 4] - arr[:, 1]) * arr[:, 5]
    elif transform_str == 'C*V':
        out_arr = arr[:, 4] * arr[:, 5]
    else:
        print(f'ERRO. a tranformação {transform_str} não está implementada')
        exit(-1)

    out_arr = np.reshape(out_arr, (len(out_arr), 1))
    return out_arr


def transform_directory(directory: str, transform_str: str):
    print(f'transformando diretório {directory}')

    setup = read_json('settings.json')
    timeframe = setup['timeframe']
    hist = HistMulti(directory, timeframe)

    for _symbol in hist.symbols:
        # print(_symbol)
        _symbol_timeframe = f'{_symbol}_{hist.timeframe}'
        arr = hist.arr[_symbol_timeframe]
        data = apply_transform_str(arr, transform_str)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[:, 0], True)
        dataf.columns = range(dataf.columns.size)
        _filepath = hist.get_csv_filepath(_symbol_timeframe)
        dataf.to_csv(_filepath, index=False, sep='\t')

    print(f'todos os símbolos do diretório {directory} foram transformados: {transform_str}.')


def transform_directory_(directory: str, transform_str: str):
    print(f'transformando diretório {directory}')

    setup = read_json('settings.json')
    timeframe = setup['timeframe']

    hist = HistMulti(directory, timeframe)

    if transform_str == '(C-O)*V':
        for _symbol in hist.symbols:
            # print(_symbol)
            _symbol_timeframe = f'{_symbol}_{hist.timeframe}'
            arr = hist.arr[_symbol_timeframe]
            data = (arr[:, 4] - arr[:, 1]) * arr[:, 5]
            data = np.reshape(data, (len(data), 1))
            dataf = pd.DataFrame(data)
            dataf.insert(0, 0, arr[:, 0], True)
            dataf.columns = range(dataf.columns.size)
            _filepath = hist.get_csv_filepath(_symbol_timeframe)
            dataf.to_csv(_filepath, index=False, sep='\t')
    elif transform_str == 'C*V':
        for _symbol in hist.symbols:
            # print(_symbol)
            _symbol_timeframe = f'{_symbol}_{hist.timeframe}'
            arr = hist.arr[_symbol_timeframe]
            data = arr[:, 4] * arr[:, 5]
            data = np.reshape(data, (len(data), 1))
            dataf = pd.DataFrame(data)
            dataf.insert(0, 0, arr[:, 0], True)
            dataf.columns = range(dataf.columns.size)
            _filepath = hist.get_csv_filepath(_symbol_timeframe)
            dataf.to_csv(_filepath, index=False, sep='\t')
    else:
        print(f'ERRO. a tranformação {transform_str} não está implementada')
        exit(-1)

    print(f'todos os símbolos do diretório {directory} foram transformados: {transform_str}.')


def transform_files(filepath_list: list[str], directory: str, transform_str: str):
    print(f'transformando alguns símbolos do diretório {directory}')

    if transform_str == '(C-O)*V':
        for _filepath in filepath_list:
            # print(_filepath)
            df: pd.DataFrame = pd.read_csv(_filepath, sep='\t')
            arr = df.to_numpy()
            data = (arr[:, 4] - arr[:, 1]) * arr[:, 5]
            data = np.reshape(data, (len(data), 1))
            dataf = pd.DataFrame(data)
            dataf.insert(0, 0, arr[:, 0], True)
            dataf.columns = range(dataf.columns.size)
            dataf.to_csv(_filepath, index=False, sep='\t')
    else:
        print(f'ERRO. a tranformação {transform_str} não está implementada')
        exit(-1)

    print(f'{len(filepath_list)} símbolos do diretório {directory} foram transformados: {filepath_list}.')


def transform_symbols(hist: HistMulti, transform_str: str, symbols: list[str] = None):
    print(f'transformando símbolos do objeto hist {type(hist)}')

    for symbol in hist.symbols:
        if symbols and symbol not in symbols:
            continue

        # print(symbol)
        _symbol_timeframe = f'{symbol}_{hist.timeframe}'

        arr = hist.arr[_symbol_timeframe]
        data = apply_transform_str(arr, transform_str)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[:, 0], True)
        dataf.columns = range(dataf.columns.size)
        hist.arr[_symbol_timeframe] = dataf.to_numpy(copy=True)

    hist.update_sheets()
    print(f'todos os símbolos de {type(hist)} foram tranformados.')
