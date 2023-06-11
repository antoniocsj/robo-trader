import pickle

import numpy as np
import pandas as pd

from Hist import Hist
from HistMulti import HistMulti
from sklearn.preprocessing import MinMaxScaler


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
        df2.at[i, '<VOL>'] = 0
        df2.at[i, '<SPREAD>'] = 0

        if i % 5000 == 0:
            print(f'{100 * i / len_df2:.2f} %')

        _close_ant = _close

    print(df2)
    df2.to_csv(filepath2, sep='\t', index=False, float_format='%.2f')


def normalize_directory():
    hist = HistMulti('../csv')
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


def denormalize__directory():
    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    hist = HistMulti('../csv')

    for _symbol in hist.symbols:
        _symbol_timeframe = f'{_symbol}_{hist.timeframe}'
        arr = hist.arr[_symbol_timeframe]
        data = arr[:, 2:7]
        trans: MinMaxScaler = scalers[_symbol_timeframe]
        data_inv = trans.inverse_transform(data)
        print(f'{_symbol} {data_inv[0]}')


if __name__ == '__main__':
    # criar_hist_csv()
    # hist = HistMulti('../csv')
    # for i in range(3, 10):
    #     entradas = formar_entradas_multi(hist, _index=i, _num_velas=3, _tipo_vela='C')
    #     print(f'index = {i} {entradas}')
    # normalize_directory()
    # denormalize__directory()
    pass
