import os
import pickle
import filecmp
import numpy as np
from numpy import ndarray
import pandas as pd
from Hist import Hist
from HistMulti import HistMulti
from sklearn.preprocessing import MinMaxScaler
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def write_json(_filename: str, _dict: dict):
    with open(_filename, 'w') as file:
        json.dump(_dict, file, indent=4)


def read_json(_filename: str) -> dict:
    if os.path.exists(_filename):
        with open(_filename, 'r') as file:
            _dict = json.load(file)
    else:
        print(f'ERRO. O arquivo {_filename} não foi encontrado.')
        exit(-1)
    return _dict


# usada na GP
def formar_entradas(arr: np.ndarray, index: int, _num_velas: int, _tipo_vela: str) -> list[float]:
    """
    Estas função se baseia num array numpy para gerar uma lista de floats.
    Dado um index, que é uma linha da tabela, retorna uma lista dos valores que compoem as
    as linhas anteriores a esse index. O número de linha anteriores que irão fornecer os valores
    OHLC (ou OHLCV, dependendo do tipo da vela) é dado por _num_velas.
    :param arr: array numpy proveniente de um arquivo csv, onde cada linha do arquivo é uma vela.
    :param index: posição da vela atual
    :param _num_velas: quantas velas anteriores vou coletar
    :param _tipo_vela: uma string semelhante a OHLC ou OHLCV, que decidirá se o volume da vela será incluído.
    :return: uma lista do tipo list[floats] comum do python contendo as _num_velas anteriores a vela que está em index.
    """
    _entradas = []

    if _tipo_vela == 'OHLCV':
        col_final = len(_tipo_vela) + 2
        for vela in arr[index - _num_velas:index]:
            _entradas += vela[2:col_final].tolist()
    elif _tipo_vela == 'OHLC':
        col_final = len(_tipo_vela) + 2
        for vela in arr[index - _num_velas:index]:
            _entradas += vela[2:col_final].tolist()
    elif _tipo_vela == 'CV':
        for vela in arr[index - _num_velas:index]:
            _entradas += vela[5:7].tolist()
    elif _tipo_vela == 'C':
        for vela in arr[index - _num_velas:index]:
            _entradas += vela[5:6].tolist()

    return _entradas


# usada na GP
def formar_entradas_multi(_hist: HistMulti, _index: int, _num_velas: int, _tipo_vela: str) -> list[float]:
    _entradas = []
    _timeframe = _hist.timeframe

    if _tipo_vela == 'OHLC':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            for _vela in _hist.arr[_symbol_timeframe][_index - _num_velas:_index]:
                _entradas += _vela[2:6].tolist()
    elif _tipo_vela == 'OHLCV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            for _vela in _hist.arr[_symbol_timeframe][_index - _num_velas:_index]:
                _entradas += _vela[2:7].tolist()
    elif _tipo_vela == 'C':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            for _vela in _hist.arr[_symbol_timeframe][_index - _num_velas:_index]:
                _entradas += _vela[5:6].tolist()
    elif _tipo_vela == 'CV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            for _vela in _hist.arr[_symbol_timeframe][_index - _num_velas:_index]:
                _entradas += _vela[5:7].tolist()
    else:
        print(f'tipo de vela não suportado: {_tipo_vela}')
        exit(-1)

    return _entradas


# escrever uma função que cria um arquivo csv que represente um histórico fictício de um par de moeda fictício.
# as velas devem seguir um padrão simples definido por alguma função matemática (do tipo senoidal, por exemplo).
# o objetivo destes dados históricos é facilitar o estudo da aplicação da programação genética no day trading.
# os dados históricos reais produzidos pelos pares de moeda do forex são muito complexos e ruidosos, e isso complica
# a busca de padrões que possam auxiliar nas operações de day trade guiadas por algorítimos gerados por programação
# genética.
def criar_hist_csv():
    from pandas import DataFrame
    # primeiro, pegue um csv contendo um histórico real e, a partir dele, gere o fictício.
    # a vantagem de começar usando um csv com dados reais é porque já contém as colunas <DATE> e <TIME>.
    symbol1 = 'XAUUSD'
    symbol2 = 'XAUUSD-SENO'
    timeframe = 'M5'
    hist = Hist('../csv')
    hist.get_hist_data(symbol1, timeframe)

    df2: DataFrame = hist.df.copy()

    filepath1 = hist.get_csv_filepath(f'{symbol1}_{timeframe}')
    filepath2 = filepath1.replace(symbol1, symbol2)

    t = 360
    # p = 12
    p = 32
    d = t / p
    time = np.linspace(0, t - d, p)
    # data = np.sin(time*np.pi/180)
    data = np.sin(0.5 * np.pi * time * np.pi / 180) + np.cos(2 * np.pi * time * np.pi / 180) * np.cos(
        0.5 * np.pi * time * np.pi / 180)
    valor_central = 1000.0
    upper_shadow_height = 0.1
    lower_shadow_height = 0.1
    print(time)
    print(data)

    _close_ant = valor_central
    len_df2 = len(df2)
    for i in range(len_df2):
        _close = data[i % p] + valor_central
        _open = _close_ant

        if _close >= _open:
            # vela/candlestick de alta (positivo)
            _low = _open - lower_shadow_height
            _high = _close + upper_shadow_height
        else:
            # vela/candlestick de baixa (negativo)
            _low = _close - lower_shadow_height
            _high = _open + upper_shadow_height

        df2.at[i, '<HIGH>'] = _high
        df2.at[i, '<CLOSE>'] = _close
        df2.at[i, '<OPEN>'] = _open
        df2.at[i, '<LOW>'] = _low
        df2.at[i, '<TICKVOL>'] = 0
        # df2.at[i, '<VOL>'] = 0
        # df2.at[i, '<SPREAD>'] = 0

        if i % 5000 == 0:
            print(f'{100 * i / len_df2:.2f} %')

        _close_ant = _close

    print(df2)
    df2.to_csv(filepath2, sep='\t', index=False, float_format='%.2f')


def renameArguments(_pset, _num_velas: int, _tipo_vela: str):
    _arguments = {}
    k = len(_tipo_vela)

    if _tipo_vela == 'OHLCV':
        for i in range(_num_velas):
            _arguments[f'X{i * k + 0}'] = f'O{i}'
            _arguments[f'X{i * k + 1}'] = f'H{i}'
            _arguments[f'X{i * k + 2}'] = f'L{i}'
            _arguments[f'X{i * k + 3}'] = f'C{i}'
            _arguments[f'X{i * k + 4}'] = f'V{i}'
    elif _tipo_vela == 'OHLC':
        for i in range(_num_velas):
            _arguments[f'X{i * k + 0}'] = f'O{i}'
            _arguments[f'X{i * k + 1}'] = f'H{i}'
            _arguments[f'X{i * k + 2}'] = f'L{i}'
            _arguments[f'X{i * k + 3}'] = f'C{i}'
    elif _tipo_vela == 'CV':
        for i in range(_num_velas):
            _arguments[f'X{i * k + 0}'] = f'C{i}'
            _arguments[f'X{i * k + 1}'] = f'V{i}'
    elif _tipo_vela == 'C':
        for i in range(_num_velas):
            _arguments[f'X{i * k + 0}'] = f'C{i}'

    _pset.renameArguments(**_arguments)


def renameArgumentsMulti(_pset, _hist, _num_velas: int, _tipo_vela: str):
    _arguments = {}
    k = len(_tipo_vela)
    nv = _num_velas

    if _tipo_vela == 'OHLCV':
        for s in range(len(_hist.symbols)):
            _symbol = _hist.symbols[s]
            for i in range(_num_velas):
                _arguments[f'X{k * i + s * nv * k + 0}'] = f'{_symbol}_O{i}'
                _arguments[f'X{k * i + s * nv * k + 1}'] = f'{_symbol}_H{i}'
                _arguments[f'X{k * i + s * nv * k + 2}'] = f'{_symbol}_L{i}'
                _arguments[f'X{k * i + s * nv * k + 3}'] = f'{_symbol}_C{i}'
                _arguments[f'X{k * i + s * nv * k + 4}'] = f'{_symbol}_V{i}'
    elif _tipo_vela == 'OHLC':
        for s in range(len(_hist.symbols)):
            _symbol = _hist.symbols[s]
            for i in range(_num_velas):
                _arguments[f'X{k * i + s * nv * k + 0}'] = f'{_symbol}_O{i}'
                _arguments[f'X{k * i + s * nv * k + 1}'] = f'{_symbol}_H{i}'
                _arguments[f'X{k * i + s * nv * k + 2}'] = f'{_symbol}_L{i}'
                _arguments[f'X{k * i + s * nv * k + 3}'] = f'{_symbol}_C{i}'
    elif _tipo_vela == 'CV':
        for s in range(len(_hist.symbols)):
            _symbol = _hist.symbols[s]
            for i in range(_num_velas):
                _arguments[f'X{k * i + s * nv * k + 0}'] = f'{_symbol}_C{i}'
                _arguments[f'X{k * i + s * nv * k + 1}'] = f'{_symbol}_V{i}'
    elif _tipo_vela == 'C':
        for s in range(len(_hist.symbols)):
            _symbol = _hist.symbols[s]
            for i in range(_num_velas):
                _arguments[f'X{k * i + s * nv * k + 0}'] = f'{_symbol}_C{i}'

    _pset.renameArguments(**_arguments)


def are_dir_trees_equal(dir1, dir2):
    """
    Compare two directories recursively. Files in each directory are
    assumed to be equal if their names and contents are equal.

    @param dir1: First directory path
    @param dir2: Second directory path

    @return: True if the directory trees are the same and
        there were no errors while accessing the directories or files,
        False otherwise.
   """

    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if len(dirs_cmp.left_only) > 0 or len(dirs_cmp.right_only) > 0 or len(dirs_cmp.funny_files) > 0:
        return False
    (_, mismatch, errors) = filecmp.cmpfiles(
        dir1, dir2, dirs_cmp.common_files, shallow=False)
    if len(mismatch) > 0 or len(errors) > 0:
        return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = os.path.join(dir1, common_dir)
        new_dir2 = os.path.join(dir2, common_dir)
        if not are_dir_trees_equal(new_dir1, new_dir2):
            return False
    return True


def get_list_sync_files(directory: str):
    _list = []
    all_files = os.listdir(directory)

    for filename in all_files:
        if filename.startswith('sync_cp_') and filename.endswith('.json'):
            _list.append(filename)

    return sorted(_list)


def load_sync_cp_file(_dirname: str, _filename: str) -> dict:
    _filepath = f'{_dirname}/{_filename}'
    if os.path.exists(_filepath):
        with open(_filepath, 'r') as file:
            cp = json.load(file)
    else:
        print(f'erro em load_sync_file(). arquivo {_filepath} não foi encontrado.')
        exit(-1)
    return cp


def search_symbols(directory: str, timeframe: str):
    """
    Procurando pelos símbolos nos nomes dos arquivos csv.
    Considera erro encontrar símbolos repetidos ou mais de 1 timeframe.
    :return: lista dos símbolos ordenada alfabeticamente e um dicionário contendo os caminhos relativos
             para os arquivos CSVs correspondentes.
    """
    # passe por todos os arquivos csv e descubra o symbol e timeframe
    if not os.path.exists(directory):
        print(f'ERRO. o diretório {directory} não existe.')
        exit(-1)

    symbols_names = []
    timeframes = set()
    symbols_paths = {}

    all_files = os.listdir(directory)
    for filename in all_files:
        if filename.endswith('.csv'):
            _symbol = filename.split('_')[0]
            _timeframe = filename.split('_')[1]
            if _timeframe.endswith('.csv'):
                _timeframe = _timeframe.replace('.csv', '')

            if _symbol not in symbols_names:
                symbols_names.append(_symbol)
                _filepath = f'{directory}/{filename}'
                symbols_paths[f'{_symbol}_{_timeframe}'] = _filepath
                timeframes.add(_timeframe)
            else:
                print(f'ERRO. o símbolo {_symbol} aparece repetido no mesmo diretório')
                exit(-1)

    if len(timeframes) > 1:
        print(f'ERRO. Há mais de 1 timeframe no diretório {directory}')
        exit(-1)

    if list(timeframes)[0] != timeframe:
        print(f'ERRO. O diretório {directory} não possui símbolos com timeframe {timeframe}')
        exit(-1)

    symbols_names = sorted(symbols_names)
    return symbols_names, symbols_paths


def calc_n_inputs(directory: str, tipo_vela: str, timeframe: str):
    """
    Faz uma varredura no diretório e retorna o número de colunas (além da data e horário) que há em cada arquivos
    CSV e também retorna o número de símbolos/arquivos CSVs
    :param directory:
    :param tipo_vela:
    :param timeframe:
    :return:
    """
    count = 0
    symbols_names, symbols_paths = search_symbols(directory, timeframe)
    for s in symbols_names:
        if s.endswith('@T') or s.endswith('@DT'):
            count += 1
        else:
            count += len(tipo_vela)
    return count, len(symbols_names)


def normalize_directory(directory: str):
    hist = HistMulti(directory)
    scalers = {}

    for _symbol in hist.symbols:
        print(_symbol)
        _symbol_timeframe = f'{_symbol}_{hist.timeframe}'
        arr = hist.arr[_symbol_timeframe]
        data = arr[:, 2:7]
        trans = MinMaxScaler()
        data = trans.fit_transform(data)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[:, 0], True)
        dataf.insert(1, 1, arr[:, 1], True)
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

    hist = HistMulti(directory)

    for _symbol in hist.symbols:
        _symbol_timeframe = f'{_symbol}_{hist.timeframe}'
        arr = hist.arr[_symbol_timeframe]
        data = arr[:, 2:7]
        trans: MinMaxScaler = scalers[_symbol_timeframe]
        data_inv = trans.inverse_transform(data)
        print(f'{_symbol} {data_inv[0]}')

    print(f'todos os símbolos do diretório {directory} foram desnormalizados.')


def differentiate_directory(directory: str):
    print(f'diferenciando diretório {directory}')
    hist = HistMulti(directory)

    for _symbol in hist.symbols:
        print(_symbol)
        _symbol_timeframe = f'{_symbol}_{hist.timeframe}'
        arr = hist.arr[_symbol_timeframe]
        data = arr[:, 2:7]
        data = np.diff(data, axis=0)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[1:, 0], True)
        dataf.insert(1, 1, arr[1:, 1], True)
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
        data = arr[:, 2:7]
        data = np.diff(data, axis=0)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[1:, 0], True)
        dataf.insert(1, 1, arr[1:, 1], True)
        dataf.columns = range(dataf.columns.size)
        dataf.to_csv(_filepath, index=False, sep='\t')

    print(f'{len(filepath_list)} símbolos do diretório {directory} foram diferenciados: {filepath_list}')


def transform_directory(directory: str, transform_str: str):
    print(f'transformando diretório {directory}')
    hist = HistMulti(directory)

    if transform_str == '(C-O)*V':
        for _symbol in hist.symbols:
            print(_symbol)
            _symbol_timeframe = f'{_symbol}_{hist.timeframe}'
            arr = hist.arr[_symbol_timeframe]
            data = (arr[:, 5] - arr[:, 2]) * arr[:, 6]
            data = np.reshape(data, (len(data), 1))
            dataf = pd.DataFrame(data)
            dataf.insert(0, 0, arr[:, 0], True)
            dataf.insert(1, 1, arr[:, 1], True)
            dataf.columns = range(dataf.columns.size)
            _filepath = hist.get_csv_filepath(_symbol_timeframe)
            dataf.to_csv(_filepath, index=False, sep='\t')
    elif transform_str == 'C*V':
        for _symbol in hist.symbols:
            print(_symbol)
            _symbol_timeframe = f'{_symbol}_{hist.timeframe}'
            arr = hist.arr[_symbol_timeframe]
            data = arr[:, 5] * arr[:, 6]
            data = np.reshape(data, (len(data), 1))
            dataf = pd.DataFrame(data)
            dataf.insert(0, 0, arr[:, 0], True)
            dataf.insert(1, 1, arr[:, 1], True)
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
            data = (arr[:, 5] - arr[:, 2]) * arr[:, 6]
            data = np.reshape(data, (len(data), 1))
            dataf = pd.DataFrame(data)
            dataf.insert(0, 0, arr[:, 0], True)
            dataf.insert(1, 1, arr[:, 1], True)
            dataf.columns = range(dataf.columns.size)
            dataf.to_csv(_filepath, index=False, sep='\t')
    else:
        print(f'ERRO. a tranformação {transform_str} não está implementada')
        exit(-1)

    print(f'{len(filepath_list)} símbolos do diretório {directory} foram transformados: {filepath_list}.')


def denorm_close_price(_c, trans: MinMaxScaler):
    c_denorm = trans.inverse_transform(np.array([0, 0, 0, _c, 0], dtype=object).reshape(1, -1))
    c_denorm = c_denorm[0][3]
    return c_denorm


def save_train_configs(_train_configs: dict):
    with open("train_configs.json", "w") as file:
        json.dump(_train_configs, file, indent=4, sort_keys=False, cls=NpEncoder)


# usada nas criação de amostra de treinamento das redes neurais
def prepare_train_data_multi_0(_hist: HistMulti, _symbol_out: str, _start_index: int,
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


# usada nas criação de amostra de treinamento das redes neurais
def prepare_train_data_multi(_hist: HistMulti, _symbol_out: str, _start_index: int,
                             _num_velas: int, _tipo_vela: str) -> ndarray:
    _data = []
    _timeframe = _hist.timeframe
    _symbol_tf_out = f'{_symbol_out}_{_timeframe}'

    if _tipo_vela == 'C':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 3:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 2]
            else:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 5]
            _data_in = _data_in.reshape(len(_data_in), 1)
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 5]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _tipo_vela == 'CV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 3:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 2]
                _data_in = _data_in.reshape(len(_data_in), 1)
            else:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 5:7]
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 5]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _tipo_vela == 'OHLC':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 3:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 2]
                _data_in = _data_in.reshape(len(_data_in), 1)
            else:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 2:6]
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))

        _c_out = _hist.arr[_symbol_tf_out][_start_index + 1:_start_index + _num_velas + 1][:, 5]
        _c_out = _c_out.reshape(len(_c_out), 1)
        _data = np.hstack((_data, _c_out))
    elif _tipo_vela == 'OHLCV':
        for _symbol in _hist.symbols:
            _symbol_timeframe = f'{_symbol}_{_timeframe}'
            if _hist.arr[_symbol_timeframe].shape[1] == 3:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 2]
                _data_in = _data_in.reshape(len(_data_in), 1)
            else:
                _data_in = _hist.arr[_symbol_timeframe][_start_index:_start_index + _num_velas][:, 2:7]
            if len(_data) == 0:
                _data = _data_in
            else:
                _data = np.hstack((_data, _data_in))

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


if __name__ == '__main__':
    pass
