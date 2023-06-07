import os
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame


class HistMulti:
    def __init__(self, directory: str):
        self.directory = directory  # diretório onde se encontra os arquivos csv
        self.all_files = []
        self.csv_files = {}  # guarda os nomes dos arquivos csv conforme seu 'simbolo' e 'timeframe'
        self.symbols = []
        self.timeframe = ''
        self.hist_csv = {}  # guarda os nomes dos arquivos csv conforme seu 'simbolo' e 'timeframe'
        self.arr = {}  # guarda os arrays dos dados históricos conforme seu 'simbolo' e 'timeframe'

        self.search_symbols()
        self.load_symbols()

    def search_symbols(self):
        """
        Procurando pelos arquivos csv correspondentes ao 'simbolo' e ao 'timeframe'
        :return:
        """
        # passe por todos os arquivos csv e descubra o symbol e timeframe
        self.all_files = os.listdir(self.directory)
        for filename in self.all_files:
            if filename.endswith('.csv'):
                _symbol = filename.split('_')[0]
                _timeframe = filename.split('_')[1]
                if _timeframe.endswith('.csv'):
                    _timeframe = _timeframe.replace('.csv', '')

                if _symbol not in self.symbols:
                    self.symbols.append(_symbol)
                else:
                    print(f'erro. o símbolo {_symbol} aparece repetido no mesmo diretório')
                    exit(-1)

                if self.timeframe == '':
                    self.timeframe = _timeframe
                elif _timeframe != self.timeframe:
                    print(f'erro. há mais de um timeframe no mesmo diretório')
                    exit(-1)

                self.csv_files[f'{_symbol}_{_timeframe}'] = filename

        self.symbols = sorted(self.symbols)

    def get_csv_filepath(self, _symbol_timeframe: str) -> str:
        _filepath = self.directory + '/' + self.csv_files[_symbol_timeframe]
        return _filepath

    def add_hist_data(self, _symbol: str, _timeframe: str):
        _filepath = self.get_csv_filepath(f'{_symbol}_{_timeframe}')
        key = f'{_symbol}_{_timeframe}'

        df: DataFrame = pd.read_csv(_filepath, delimiter='\t')
        # df.drop(columns=['<VOL>', '<SPREAD>'], inplace=True)
        if df.isnull().sum().values.sum() != 0:
            print(f'Há dados faltando no arquivo {_filepath}')
            exit(-1)

        self.arr[key] = df.to_numpy(copy=True)
        # print(f'{key} carregando dados a partir de {_filepath}. {len(self.arr[key])} linhas')
        del df

    def load_symbols(self):
        for s in self.symbols:
            self.add_hist_data(s, self.timeframe)

    def print_hist(self):
        print('primeiras linhas dos históricos:')
        if len(self.arr) > 0:
            for k, v in enumerate(self.arr):
                print(k, v, self.arr[v][0])
        else:
            print('os dados históricos não foram carregados ainda.')


if __name__ == '__main__':
    hist = HistMulti('../csv')
    # list_assets = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDUSD', 'CADCHF', 'CADJPY', 'CHFJPY',
    #                'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURUSD', 'GBPAUD',
    #                'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY',
    #                'XAUUSD']

    hist.print_hist()
