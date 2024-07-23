from typing import Any
import os
from numpy import ndarray
import numpy as np
import pandas as pd
import pickle
from src.HistMulti import HistMulti
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.utils.utils_filesystem import read_json


def normalize_minmax(data: ndarray, feature_range: tuple, csv_content=None) -> tuple[ndarray, MinMaxScaler]:
    """
    O MinMaxScaler normaliza uma matrix considerando que cada coluna é completamente independente das outras.
    Assim ele calcula o máximo e mínimo de cada coluna separadamente.
    Porém, é possível adaptar o MinMaxScaler para que o seu uso seja diferente.

    1) supondo que todas as colunas possuem dados da mesma variável/característca/fenômeno e com a mesma unidade de medida.
    Nesse caso, seria mais apropriado o cálculo do máximo e mínimo globais da matriz.
    Por exemplo, uma matrix OHLC em que todas as colunas se referem a mesma variável=preço.
    Nesse caso, csv_content = 'HOMOGENEOUS'.

    2) supondo que as algumas colunas possuem dados da mesma variável/característca/fenômeno e com a mesma unidade de medida e
    que outras coluna sejam de outras váriáveis. Porém a última, possuem uma unidade de medida diferente.
    Nesse caso, seria mais apropriado o cálculo dos máximos e mínimos regionais da matriz.
    Por exemplo, uma matriz OHLCV em que as 4 primeiras colunas são do preço e a 5ª coluna é do volume.
    Assim, as colunas do preço terão um máximo/mínimo regional e a coluna do volume terá seu próprio máximo/mínimo regional.
    Nestes casos, csv_content = 'HETEROGENEOUS_OHLCV'.

    3) caso padrão, em que cada coluna é completamente independente das outras.
    Nesse caso, csv_content = 'HETEROGENEOUS_DEFAULT'.


    Parameters
    ----------
    data: um array do tipo ndarray.
    feature_range: será repassada ao MinMaxScaler
    csv_content: 'HETEROGENEOUS_DEFAULT' ou 'HETEROGENEOUS_OHLCV' ou 'HOMOGENEOUS'.

    Returns ndarray normalizado
    -------

    """
    if not csv_content or csv_content.upper() == 'HETEROGENEOUS_DEFAULT':
        # saída padrão do MinMaxScaler()
        scaler = MinMaxScaler(feature_range=feature_range)
        return scaler.fit_transform(data), scaler
    elif csv_content.upper() == 'HOMOGENEOUS':  # solução do problema 1
        data2 = np.array([[data.min()] * data.shape[1], [data.max()] * data.shape[1]])
        scaler = MinMaxScaler(feature_range=feature_range)
        scaler.fit(data2)
        return scaler.transform(data), scaler
    elif csv_content.upper() == 'HETEROGENEOUS_OHLCV':  # solução do problema 2
        if data.shape[1] != 5:
            print(f'ERRO. O array não segue o formato HETEROGENEOUS_OHLCV que exige 5 colunas. '
                  f'data.shape[1] = {data.shape[1]}')
            exit(-1)
        data2a = data[:, 0:4]
        data2b = data[:, 4]
        min_ohlc = data2a.min()
        max_ohlc = data2a.max()
        min_v = data2b.min()
        max_v = data2b.max()
        data3 = np.array([[min_ohlc] * 4 + [min_v],
                          [max_ohlc] * 4 + [max_v]])
        scaler = MinMaxScaler(feature_range=feature_range)
        scaler.fit(data3)
        return scaler.transform(data), scaler
    else:
        print(f'ERRO! csv_content = {csv_content} não suportado')
        exit(-1)


def normalize_standard(data: ndarray, data_format=None) -> tuple[ndarray, StandardScaler]:
    """
    Leia o comentário da função normalize_minmax.
    Parameters
    ----------
    data: um array do tipo ndarray.
    data_format: 'HETEROGENEOUS_DEFAULT' ou 'HETEROGENEOUS_OHLCV' ou 'HOMOGENEOUS'.

    Returns ndarray normalizado
    -------

    """
    if not data_format or data_format.upper() == 'HETEROGENEOUS_DEFAULT':
        # saída padrão do StandardScaler()
        scaler = StandardScaler()
        return scaler.fit_transform(data), scaler
    elif data_format.upper() == 'HOMOGENEOUS':  # solução do problema 1
        scaler = StandardScaler()
        scaler.fit(data)
        scaler.mean_ = np.array([np.mean(data)] * data.shape[1])
        scaler.var_ = np.array([np.var(data)] * data.shape[1])
        scaler.scale_ = np.sqrt(scaler.var_)
        return scaler.transform(data), scaler

    elif data_format.upper() == 'HETEROGENEOUS_OHLCV':  # solução do problema 2
        if data.shape[1] != 5:
            print(f'ERRO. O array não segue o formato HETEROGENEOUS_OHLCV que exige 5 colunas. '
                  f'data.shape[1] = {data.shape[1]}')
            exit(-1)
        scaler = StandardScaler()
        scaler.fit(data)
        data_ohlc = data[:, 0:4]
        data_v = data[:, 4]

        mean_ohlc = data_ohlc.mean()
        var_ohlc = data_ohlc.var()
        mean_v = data_v.mean()
        var_v = data_v.var()

        scaler.mean_ = np.array([mean_ohlc] * 4 + [mean_v])
        scaler.var_ = np.array([var_ohlc] * 4 + [var_v])
        scaler.scale_ = np.sqrt(scaler.var_)

        return scaler.transform(data), scaler

    else:
        print(f'ERRO! csv_content = {data_format} não suportado')
        exit(-1)


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
    working_dir = os.getcwd()
    settings_filepath = os.path.join(working_dir, 'settings.json')
    settings = read_json(settings_filepath)

    timeframe = settings['timeframe']
    scaler_feature_range = tuple(settings['scaler_feature_range'])
    csv_content = settings['csv_content']
    normalization_method = settings['normalization_method']

    print(f'características da normalização: csv_content = {csv_content}, normalization_method = {normalization_method}')

    hist = HistMulti(directory, timeframe)
    scalers = {}

    for symbol in hist.symbols:
        arr: ndarray = hist.arr[symbol][timeframe]
        if arr.shape[1] == 2:
            data: ndarray = arr[:, 1]
            data = data.reshape(len(data), 1)
        else:
            data = arr[:, 1:6]
        
        # scaler = MinMaxScaler(feature_range=scaler_feature_range)
        # data = scaler.fit_transform(data)
        if normalization_method == 'minmax':
            data, scaler = normalize_minmax(data, feature_range=scaler_feature_range, csv_content=csv_content)
        elif normalization_method == 'standard':
            data, scaler = normalize_standard(data, data_format=csv_content)
        else:
            print(f'ERRO! normalization_method = {normalization_method} inválido.')
            exit(-1)

        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[:, 0], True)
        dataf.columns = range(dataf.columns.size)
        _filepath = hist.get_csv_filepath(symbol, timeframe)
        dataf.to_csv(_filepath, index=False, sep='\t')
        _symbol_timeframe = f'{symbol}_{timeframe}'
        scalers[_symbol_timeframe] = scaler

    # guarda scalers.pkl no próprio diretório onde estão os símbolos (ou históricos/csv)
    scalers_filepath = os.path.join(directory, 'scalers.pkl')
    with open(scalers_filepath, 'wb') as file:
        pickle.dump(scalers, file)

    print(f'todos os símbolos do diretório {directory} foram normalizados.')
    print('o arquivo scalers.pkl, que guarda as informações da normalização, foi salvo.')
    print(f'conteúdo de scalers.pkl:')
    print(f'{scalers}')


def normalize_symbols(hist: HistMulti, scalers: dict, symbols: list[str] = None):
    timeframe = hist.timeframe

    for symbol in hist.symbols:
        if symbols and symbol not in symbols:
            continue

        # print(symbol)
        _symbol_timeframe = f'{symbol}_{timeframe}'

        arr: ndarray = hist.arr[symbol][timeframe]
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
        hist.arr[symbol][timeframe] = dataf.to_numpy(copy=True)

    if symbols:
        # print(f'os símbolos {symbols} de {type(hist)} foram normalizados.')
        hist.update_sheets(symbols)
    else:
        hist.update_sheets()
        # print(f'todos os símbolos de {type(hist)} foram normalizados.')


def differentiate_directory(directory: str):
    print(f'diferenciando diretório {directory}')

    setup = read_json('../settings.json')
    timeframe = setup['timeframe']

    hist = HistMulti(directory, timeframe)

    for symbol in hist.symbols:
        # print(symbol)

        arr = hist.arr[symbol][timeframe]
        if arr.shape[1] == 2:
            data = arr[:, 1]
        else:
            data = arr[:, 1:6]

        data = np.diff(data, axis=0)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[1:, 0], True)
        dataf.columns = range(dataf.columns.size)
        _filepath = hist.get_csv_filepath(symbol, timeframe)
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

    timeframe = hist.timeframe

    for symbol in hist.symbols:
        if symbols and symbol not in symbols:
            continue

        # print(symbol)
        arr = hist.arr[symbol][timeframe]
        if arr.shape[1] == 2:
            data: ndarray = arr[:, 1]
            data = data.reshape(len(data), 1)
        else:
            data = arr[:, 1:6]

        data = np.diff(data, axis=0)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[1:, 0], True)
        dataf.columns = range(dataf.columns.size)
        hist.arr[symbol][timeframe] = dataf.to_numpy(copy=True)

    if symbols:
        # print(f'os símbolos {symbols} de {type(hist)} foram diferenciados.')
        hist.update_sheets(symbols)
    else:
        hist.update_sheets()
        # print(f'todos os símbolos de {type(hist)} foram diferenciados.')


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

    setup = read_json('../settings.json')
    timeframe = setup['timeframe']
    hist = HistMulti(directory, timeframe)

    for symbol in hist.symbols:
        # print(symbol)
        arr = hist.arr[symbol][timeframe]
        data = apply_transform_str(arr, transform_str)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[:, 0], True)
        dataf.columns = range(dataf.columns.size)
        _filepath = hist.get_csv_filepath(symbol, timeframe)
        dataf.to_csv(_filepath, index=False, sep='\t')

    print(f'todos os símbolos do diretório {directory} foram transformados: {transform_str}.')


def transform_directory_(directory: str, transform_str: str):
    print(f'transformando diretório {directory}')

    setup = read_json('../settings.json')
    timeframe = setup['timeframe']

    hist = HistMulti(directory, timeframe)

    if transform_str == '(C-O)*V':
        for symbol in hist.symbols:
            # print(symbol)
            arr = hist.arr[symbol][timeframe]
            data = (arr[:, 4] - arr[:, 1]) * arr[:, 5]
            data = np.reshape(data, (len(data), 1))
            dataf = pd.DataFrame(data)
            dataf.insert(0, 0, arr[:, 0], True)
            dataf.columns = range(dataf.columns.size)
            _filepath = hist.get_csv_filepath(symbol, timeframe)
            dataf.to_csv(_filepath, index=False, sep='\t')
    elif transform_str == 'C*V':
        for symbol in hist.symbols:
            # print(symbol)
            arr = hist.arr[symbol][timeframe]
            data = arr[:, 4] * arr[:, 5]
            data = np.reshape(data, (len(data), 1))
            dataf = pd.DataFrame(data)
            dataf.insert(0, 0, arr[:, 0], True)
            dataf.columns = range(dataf.columns.size)
            _filepath = hist.get_csv_filepath(symbol, timeframe)
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

    timeframe = hist.timeframe

    for symbol in hist.symbols:
        if symbols and symbol not in symbols:
            continue

        # print(symbol)
        _symbol_timeframe = f'{symbol}_{timeframe}'

        arr = hist.arr[symbol][timeframe]
        data = apply_transform_str(arr, transform_str)
        dataf = pd.DataFrame(data)
        dataf.insert(0, 0, arr[:, 0], True)
        dataf.columns = range(dataf.columns.size)
        hist.arr[symbol][timeframe] = dataf.to_numpy(copy=True)

    hist.update_sheets()
    # print(f'todos os símbolos de {type(hist)} foram tranformados.')
