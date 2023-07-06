import os
import csv
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from Sheet import Sheet


class HistMulti:
    def __init__(self, source: Any, timeframe: str):
        """
        Cria um objeto que armazena os dados históricos obtidos por:
         1) a partir de um diretório contendo arquivos CSVs ou ;
         2) a partir de uma lista de objetos Sheets (Planilhas);
        :param source: filepath (str) ou list[Sheet]
        """
        self.symbols = []
        self.timeframe = timeframe
        self.arr = {}  # guarda os arrays dos dados históricos conforme seu 'simbolo' e 'timeframe'

        if isinstance(source, str):
            self.directory = source  # diretório onde se encontra os arquivos csv
            print(f'obtendo dados históricos a partir do diretório {self.directory}')
            self.all_files = []
            self.csv_files = {}  # guarda os nomes dos arquivos csv conforme seu 'simbolo' e 'timeframe'
            self.hist_csv = {}  # guarda os nomes dos arquivos csv conforme seu 'simbolo' e 'timeframe'
        elif isinstance(source, list):
            print(f'obtendo dados históricos a partir de uma lista de planilhas')
            self.sheets: list[Sheet] = source

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

                if _timeframe != self.timeframe:
                    print(f'ERRO. o timeframe encontrado ({_timeframe}) é diferente do timeframe '
                          f'especificado ({self.timeframe})')
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


class HistMultiOriginal:
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
    from utils import read_json

    setup = read_json('setup.json')
    print(f'setup.json: {setup}')

    csv_dir = setup['csv_dir']
    symbol_out = setup['symbol_out']
    _tf = setup['timeframe']

    hist = HistMulti(csv_dir, _tf)
    hist.print_hist()
