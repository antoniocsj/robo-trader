import numpy as np


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
    col_final = len(_tipo_vela) + 2
    _entradas = []
    for vela in arr[index - _num_velas:index]:
        _entradas += vela[2:col_final].tolist()

    return _entradas
