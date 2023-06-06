import os
import csv
import numpy as np
import pandas as pd


class Hist:
    def __init__(self, directory: str):
        self.directory = directory  # diretório onde se encontra os arquivos csv
        self.all_files = []
        self.csv_files = {}  # guarda os nomes dos arquivos csv conforme seu 'simbolo' e 'timeframe'
        self.symbols = []
        self.timeframe = ''
        self.df = None
        self.arr = None

        self.search_symbols()

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

    def get_csv_filepath(self, _symbol_timeframe: str) -> str:
        _filepath = self.directory + '/' + self.csv_files[_symbol_timeframe]
        return _filepath

    def get_hist_data(self, _symbol: str, _timeframe: str):
        _filepath = self.get_csv_filepath(f'{_symbol}_{_timeframe}')
        self.df = pd.read_csv(_filepath, delimiter='\t')
        self.df.drop(columns=['<VOL>', '<SPREAD>'], inplace=True)
        # self.df.columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL']
        self.arr = self.df.to_numpy(copy=True)

        if self.df.isnull().sum().values.sum() != 0:
            print(f'Há dados faltando no arquivo {_filepath}')
            exit(-1)

    def print_hist(self):
        if self.df is None:
            print('o histórico não foi carregado ainda.')
        else:
            print(self.df)


if __name__ == '__main__':
    hist = Hist('../csv')
    hist.get_hist_data('XAUUSD', 'M5')
    hist.print_hist()

