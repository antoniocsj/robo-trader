import os
import csv
import numpy as np
import pandas as pd


class Hist:
    dir_csv = '../csv'

    def __init__(self):
        self.hist_csv = {}  # guarda os nomes dos arquivos csv conforme seu 'simbolo' e 'timeframe'
        self.df = None
        self.arr = None

        # Obtendo a lista de arquivos no diretório CSV
        self.arquivos_csv = os.listdir(Hist.dir_csv)
        self.procurar_simbolos()

    def procurar_simbolos(self):
        """
        Procurando pelos arquivos csv correspondentes ao 'simbolo' e ao 'timeframe'
        :return:
        """
        for arquivo in self.arquivos_csv:
            if arquivo.endswith('.csv'):
                _simbolo = arquivo.split('_')[0]
                _timeframe = arquivo.split('_')[1]
                self.hist_csv[f'{_simbolo}_{_timeframe}'] = arquivo

    def obter_historico(self, _simbolo: str, _timeframe: str):
        arquivo = Hist.dir_csv + '/' + self.hist_csv[f'{_simbolo}_{_timeframe}']
        self.df = pd.read_csv(arquivo, delimiter='\t')
        self.df.drop(columns=['<VOL>', '<SPREAD>'], inplace=True)
        self.df.columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL']
        self.arr = self.df.values

        if self.df.isnull().sum().values.sum() != 0:
            print(f'Há dados faltando no arquivo {arquivo}')
            exit(-1)

    def mostrar_historico(self):
        if self.df is None:
            print('o histórico não foi carregado ainda.')
        else:
            print(self.df)


if __name__ == '__main__':
    hist = Hist()
    hist.obter_historico('XAUUSD', 'H1')
    hist.mostrar_historico()

