from utils import formar_entradas_multi
from HistMulti import HistMulti
import numpy as np


def teste_1():
    dir_csv = '../csv'
    hist = HistMulti(directory=dir_csv)
    num_ativos = len(hist.symbols)
    num_velas_anteriores = 2
    tipo_vela = 'CV'
    symbol_out = 'XAUUSD'
    timeframe = hist.timeframe
    num_entradas = num_velas_anteriores * len(tipo_vela) * num_ativos
    candlesticks_quantity = 10  # quantidade de velas usadas no treinamento
    index_inicio = num_velas_anteriores
    index_inicio = 1
    index_final = index_inicio + candlesticks_quantity

    for i in range(index_inicio, index_final):
        entradas = np.array(formar_entradas_multi(hist, i, 1, tipo_vela))
        saida = np.array([hist.arr[f'{symbol_out}_{timeframe}'][i][5]])


if __name__ == '__main__':
    teste_1()
