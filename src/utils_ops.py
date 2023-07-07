import numpy as np
import pandas as pd
import pickle
from HistMulti import HistMulti
from sklearn.preprocessing import MinMaxScaler
from utils_filesystem import read_json


def denorm_close_price(_c, trans: MinMaxScaler):
    c_denorm = trans.inverse_transform(np.array([0, 0, 0, _c, 0], dtype=object).reshape(1, -1))
    c_denorm = c_denorm[0][3]
    return c_denorm


def normalize_directory(directory: str):
    setup = read_json('setup.json')
    timeframe = setup['timeframe']

    hist = HistMulti(directory, timeframe)
    scalers = {}

    for _symbol in hist.symbols:
        print(_symbol)
        _symbol_timeframe = f'{_symbol}_{hist.timeframe}'
        arr = hist.arr[_symbol_timeframe]
        data = arr[:, 1:6]
        trans = MinMaxScaler()
        data = trans.fit_transform(data)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[:, 0], True)
        dataf.columns = range(dataf.columns.size)
        _filepath = hist.get_csv_filepath(_symbol_timeframe)
        dataf.to_csv(_filepath, index=False, sep='\t')
        scalers[_symbol_timeframe] = trans

    with open('scalers.pkl', 'wb') as file:
        pickle.dump(scalers, file)

    print(f'todos os símbolos do diretório {directory} foram normalizados.')
    print('o arquivo scalers.pkl, que guarda as informações da normalização, foi salvo.')


def denormalize__directory(directory: str):
    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    setup = read_json('setup.json')
    timeframe = setup['timeframe']

    hist = HistMulti(directory, timeframe)

    for _symbol in hist.symbols:
        _symbol_timeframe = f'{_symbol}_{hist.timeframe}'
        arr = hist.arr[_symbol_timeframe]
        data = arr[:, 1:6]
        trans: MinMaxScaler = scalers[_symbol_timeframe]
        data_inv = trans.inverse_transform(data)
        print(f'{_symbol} {data_inv[0]}')

    print(f'todos os símbolos do diretório {directory} foram desnormalizados.')


def differentiate_directory(directory: str):
    print(f'diferenciando diretório {directory}')

    setup = read_json('setup.json')
    timeframe = setup['timeframe']

    hist = HistMulti(directory, timeframe)

    for _symbol in hist.symbols:
        print(_symbol)
        _symbol_timeframe = f'{_symbol}_{hist.timeframe}'
        arr = hist.arr[_symbol_timeframe]
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
        print(_filepath)
        df: pd.DataFrame = pd.read_csv(_filepath, sep='\t')
        arr = df.to_numpy()
        data = arr[:, 1:6]
        data = np.diff(data, axis=0)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[1:, 0], True)
        dataf.columns = range(dataf.columns.size)
        dataf.to_csv(_filepath, index=False, sep='\t')

    print(f'{len(filepath_list)} símbolos do diretório {directory} foram diferenciados: {filepath_list}')


def transform_directory(directory: str, transform_str: str):
    print(f'transformando diretório {directory}')

    setup = read_json('setup.json')
    timeframe = setup['timeframe']

    hist = HistMulti(directory, timeframe)

    if transform_str == '(C-O)*V':
        for _symbol in hist.symbols:
            print(_symbol)
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
            print(_symbol)
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
            print(_filepath)
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
