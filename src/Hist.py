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
        self.files_csv = os.listdir(Hist.dir_csv)
        self.search_symbols()

    def search_symbols(self):
        """
        Procurando pelos arquivos csv correspondentes ao 'simbolo' e ao 'timeframe'
        :return:
        """
        for arquivo in self.files_csv:
            if arquivo.endswith('.csv'):
                _simbolo = arquivo.split('_')[0]
                _timeframe = arquivo.split('_')[1]
                self.hist_csv[f'{_simbolo}_{_timeframe}'] = arquivo

    def get_csv_filepath(self, _simbolo: str, _timeframe: str) -> str:
        _filepath = Hist.dir_csv + '/' + self.hist_csv[f'{_simbolo}_{_timeframe}']
        return _filepath

    def get_hist_data(self, _simbolo: str, _timeframe: str):
        _filepath = self.get_csv_filepath(_simbolo, _timeframe)
        self.df = pd.read_csv(_filepath, delimiter='\t')
        # self.df.drop(columns=['<VOL>', '<SPREAD>'], inplace=True)
        # self.df.columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL']
        self.arr = self.df.to_numpy()

        if self.df.isnull().sum().values.sum() != 0:
            print(f'Há dados faltando no arquivo {_filepath}')
            exit(-1)

    def print_hist(self):
        if self.df is None:
            print('o histórico não foi carregado ainda.')
        else:
            print(self.df)


if __name__ == '__main__':
    hist = Hist()
    hist.get_hist_data('XAUUSD', 'H1')
    hist.print_hist()

