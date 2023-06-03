import os
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame


class Hist:
    dir_csv = '../csv'

    def __init__(self):
        self.hist_csv = {}  # guarda os nomes dos arquivos csv conforme seu 'simbolo' e 'timeframe'
        self.hist_data = {}  # guarda os arrays dos dados históricos conforme seu 'simbolo' e 'timeframe'

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

    def add_hist_data(self, _simbolo: str, _timeframe: str):
        _filepath = self.get_csv_filepath(_simbolo, _timeframe)
        key = f'{_simbolo}_{_timeframe}'

        df: DataFrame = pd.read_csv(_filepath, delimiter='\t')
        if df.isnull().sum().values.sum() != 0:
            print(f'Há dados faltando no arquivo {_filepath}')
            exit(-1)

        self.hist_data[key] = df.to_numpy(copy=True)

    def print_hist(self):
        if len(self.hist_data) > 0:
            for k, v in enumerate(self.hist_data):
                print(k, v, self.hist_data[v][0])
        else:
            print('os dados históricos não foram carregados ainda.')


if __name__ == '__main__':
    hist = Hist()
    # list_assets = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDUSD', 'CADCHF', 'CADJPY', 'CHFJPY',
    #                'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURUSD', 'GBPAUD',
    #                'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY',
    #                'XAUUSD']

    list_assets = ['EURUSD', 'USDJPY', 'EURGBP', 'EURJPY']
    list_assets = sorted(list_assets)

    for asset in list_assets:
        print(asset)
        hist.add_hist_data(asset, 'M5')

    hist.print_hist()

