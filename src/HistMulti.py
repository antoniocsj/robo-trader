import copy
import os
from typing import Any

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from Sheet import Sheet
from utils_filesystem import read_json
from utils_symbols import search_symbols_in_dict


class HistMulti:
    def __init__(self, source: Any, timeframe: str):
        """
        Cria um objeto que armazena os dados históricos obtidos por:
         1) a partir de um diretório contendo arquivos CSVs ou ;
         2) a partir de uma lista de objetos Sheets (Planilhas);
        :param source: filepath (str) ou dict[Sheet]
        """
        self.symbols = []
        self.timeframe = timeframe
        self.arr: dict[str, ndarray] = {}  # guarda os arrays dos dados históricos conforme seu 'simbolo' e 'timeframe'
        self.sheets: dict[str, Sheet] = {}
        self.source_is_dir = False

        if isinstance(source, str):
            self.source_is_dir = True
            self.directory = source  # diretório onde se encontra os arquivos csv
            print(f'obtendo dados históricos a partir do diretório {self.directory}')
            self.all_files = []
            self.csv_files = {}  # guarda os nomes dos arquivos csv conforme seu 'simbolo' e 'timeframe'
            self.hist_csv = {}  # guarda os nomes dos arquivos csv conforme seu 'simbolo' e 'timeframe'
            self.search_symbols()
        elif isinstance(source, dict):
            self.source_is_dir = False
            print(f'obtendo dados históricos a partir de um dicionário de planilhas')
            # self.sheets: dict[str, Sheet] = source
            self.sheets = source
            self.symbols = search_symbols_in_dict(self.sheets, self.timeframe)

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

                if self.source_is_dir:
                    self.csv_files[f'{_symbol}_{_timeframe}'] = filename

        self.symbols = sorted(self.symbols)

    def get_csv_filepath(self, _symbol_timeframe: str) -> str:
        _filepath = self.directory + '/' + self.csv_files[_symbol_timeframe]
        return _filepath

    def add_hist_data(self, symbol: str, timeframe: str):
        key = f'{symbol}_{timeframe}'
        df: DataFrame

        if self.source_is_dir:
            _filepath = self.get_csv_filepath(f'{symbol}_{timeframe}')
            df = pd.read_csv(_filepath, delimiter='\t')
            # df.drop(columns=['<VOL>', '<SPREAD>'], inplace=True)
            self.sheets[key] = Sheet(_filepath, symbol, timeframe)
        else:
            s: Sheet = self.sheets[key]
            df = s.df

        if df.isnull().sum().values.sum() != 0:
            print(f'Há dados faltando no dataframe {key}')
            exit(-1)

        self.arr[key] = df.to_numpy(copy=True)
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

    def update_sheets(self, symbols: list[str] = None):
        """
        Atualia o dicionário self.sheets para refletir os dados de self.arr
        :return:
        """
        if self.source_is_dir:
            return

        for symbol in self.symbols:
            if symbols and symbol not in symbols:
                continue
            _symbol_tf = f'{symbol}_{self.timeframe}'
            arr = self.arr[_symbol_tf]
            s: Sheet = Sheet(arr, symbol, self.timeframe)
            self.sheets[_symbol_tf] = s

    def copy_symbol(self, symbol_name_src: str, other_hist: "HistMulti", symbol_name_dst: str = None):
        if self.source_is_dir:
            print('ERRO. copy_symbol(). esta função não pode ser usada quando a fonte é um diretório.')
            exit(-1)

        symbol_tf_src = f'{symbol_name_src}_{self.timeframe}'
        if symbol_name_dst:
            symbol_tf_dst = f'{symbol_name_dst}_{self.timeframe}'
        else:
            symbol_name_dst = symbol_name_src
            symbol_tf_dst = symbol_tf_src

        if symbol_name_dst in self.symbols:
            print(f'ERRO. copy_symbol(). tentativa de adicionar o símbolo {symbol_name_dst} que já existe no hist.')
            exit(-1)

        symbol_sheet = other_hist.sheets[symbol_tf_src]
        symbols_arr = other_hist.arr[symbol_tf_src]

        self.symbols.append(symbol_name_dst)
        self.sheets[symbol_tf_dst] = copy.deepcopy(symbol_sheet)
        self.arr[symbol_tf_dst] = copy.deepcopy(symbols_arr)

    def copy_symbols(self, other_hist: "HistMulti"):
        for symbol in other_hist.symbols:
            self.copy_symbol(symbol, other_hist)

    def remove_symbol(self, symbol_name: str):
        symbol_tf = f'{symbol_name}_{self.timeframe}'

        if symbol_name not in self.symbols:
            print(f'ERRO. copy_symbol(). tentativa de remover o símbolo {symbol_name} que não existe no hist.')
            exit(-1)

        self.symbols.remove(symbol_name)
        del self.sheets[symbol_tf]
        del self.arr[symbol_tf]

    def rename_symbol(self, old_name: str, new_name: str):
        if old_name not in self.symbols:
            print(f'ERRO. rename_symbol(). tentativa de renomar o símbolo {old_name} que não existe no hist.')
            exit(-1)

        old_symbol_tf = f'{old_name}_{self.timeframe}'
        new_symbol_tf = f'{new_name}_{self.timeframe}'

        self.symbols.remove(old_name)
        self.symbols.append(new_name)
        self.sheets[new_symbol_tf] = self.sheets.pop(old_symbol_tf)
        self.arr[new_symbol_tf] = self.arr.pop(old_symbol_tf)

    def rename_symbols_adding_suffix(self, suffix: str):
        new_symbols = []
        for symbol in self.symbols[:]:
            new_symbol = f'{symbol}{suffix}'
            new_symbols.append(new_symbol)
            self.rename_symbol(symbol, new_symbol)
        self.symbols = sorted(new_symbols)

    def sort_symbols(self):
        self.symbols = sorted(self.symbols)

    def delete_first_row_symbol(self, symbol_name: str):
        symbol_tf = f'{symbol_name}_{self.timeframe}'

        if symbol_name not in self.symbols:
            print(f'ERRO. delete_first_row_symbol(). tentativa de remover primeira linha do símbolo {symbol_name} '
                  f'que não existe no hist.')
            exit(-1)

        s: Sheet = self.sheets[symbol_tf]
        df: DataFrame = s.df
        df.drop(df.index[0], inplace=True)
        df.sort_index(ignore_index=True, inplace=True)
        df.reset_index(drop=True)
        self.arr[symbol_tf] = np.delete(self.arr[symbol_tf], 0, axis=0)

    def delete_first_row_symbols(self, excepting_pattern=None):
        if excepting_pattern:
            for symbol in self.symbols:
                if excepting_pattern in symbol:
                    continue
                self.delete_first_row_symbol(symbol)
        else:
            for symbol in self.symbols:
                self.delete_first_row_symbol(symbol)

    def calc_n_features(self, candle_input_type: str):
        """
        Faz uma varredura no histórico e retorna o número de colunas (além da data e horário) que há em cada
        símbolo ou arquivo CSV.
        número de colunas = n_features.
        :param candle_input_type:
        :return: n_features (n_cols)
        """
        count = 0
        for s in self.symbols:
            if s.endswith('@T') or s.endswith('@DT') or s.endswith('@TD'):
                count += 1
            else:
                count += len(candle_input_type)
        return count


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
    setup = read_json('settings.json')
    print(f'settings.json: {setup}')

    temp_dir = setup['temp_dir']
    symbol_out = setup['symbol_out']
    _tf = setup['timeframe']

    _hist = HistMulti(temp_dir, _tf)
    _hist.print_hist()
