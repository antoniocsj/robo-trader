import numpy as np
from Hist import Hist


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
    hist = Hist()
    hist.get_hist_data(symbol1, timeframe)

    df2: DataFrame = hist.df.copy()

    filepath1 = hist.get_csv_filepath(symbol1, timeframe)
    filepath2 = filepath1.replace(symbol1, symbol2)

    t = 360
    p = 12
    d = t / p
    time = np.linspace(0, t-d, p)
    data = np.sin(time*np.pi/180)
    valor_central = 1000.0
    print(time)
    print(data)

    len_df2 = len(df2)
    for i in range(len_df2):
        y = data[i % p] + valor_central

        df2.at[i, '<HIGH>'] = y + 1
        df2.at[i, '<CLOSE>'] = y
        df2.at[i, '<OPEN>'] = y - 1
        df2.at[i, '<LOW>'] = y - 2
        df2.at[i, '<TICKVOL>'] = 0
        df2.at[i, '<VOL>'] = 0
        df2.at[i, '<SPREAD>'] = 0

        if i % 5000 == 0:
            print(f'{100*i/len_df2:.2f} %')

    print(df2)
    df2.to_csv(filepath2, sep='\t', index=False, float_format='%.2f')


if __name__ == '__main__':
    criar_hist_csv()
