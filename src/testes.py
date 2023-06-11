from utils import formar_entradas_multi
from HistMulti import HistMulti


def teste_1():
    dir_csv = '../csv'
    hist = HistMulti(directory=dir_csv)
    num_ativos = len(hist.symbols)
    num_velas_anteriores = 2
    tipo_vela = 'CV'
    num_entradas = num_velas_anteriores * len(tipo_vela) * num_ativos
    candlesticks_quantity = 500  # quantidade de velas usadas no treinamento
    index_inicio = num_velas_anteriores
    index_final = index_inicio + candlesticks_quantity

    for i in range(index_inicio, index_final):
        entradas = formar_entradas_multi(hist, i, num_velas_anteriores, tipo_vela)
        print(entradas)

if __name__ == '__main__':
    teste_1()
