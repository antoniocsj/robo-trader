from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from pandas import DataFrame
from numpy import ndarray


class Sheet:
    def __init__(self, source: Any, symbol: str, timeframe: str, csv_content: str):
        """
        Cria uma planilha a partir de um arquivo CSV ou a partir de uma list de dicionários contendo os dados para
        formar as linhas e colunas da planilha.
        :param source: filepath (str) ou list[dict]
        :param symbol: ex. EURUSD
        :param timeframe: ex. M5
        :param csv_content: conteúdo do arquivo CSV (HETEROGENEOUS_DEFAULT, HOMOGENEOUS_DEFAULT, HETEROGENEOUS_OHLCV)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.csv_content = csv_content
        self.timedelta: timedelta = self.get_timedelta()
        self.current_row = 0
        self.is_on_the_last_row = False
        self.is_trading = False

        if isinstance(source, str):
            self.filepath = source
            # print(f'criando planilha para {symbol}_{timeframe} a partir do arquivo {self.filepath}')
            self.df: DataFrame = pd.read_csv(self.filepath, delimiter='\t')
        elif isinstance(source, list):
            self.rates: list[dict] = source
            # print(f'criando planilha para {symbol}_{timeframe} a partir de uma lista')
            self.df: DataFrame = self.create_df_from_rates()
        elif isinstance(source, ndarray):
            # print(f'criando planilha para {symbol}_{timeframe} a partir de um ndarray')
            self.df: DataFrame = pd.DataFrame(source)

        if self.df.isnull().sum().values.sum() != 0:
            print(f'Há dados faltando no arquivo {self.filepath}')
            exit(-1)

        # if len(self.df.columns) == 2:
        #     self.df.columns = ['DATETIME', 'T']
        # else:
        #     self.df.columns = ['DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL']

        if csv_content == 'HETEROGENEOUS_OHLCV':
            # a primeira coluna será DATETIME e a segunda em diante será OPEN, HIGH, LOW, CLOSE, TICKVOL
            self.df.columns = ['DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICK']
        elif csv_content == 'HETEROGENEOUS_DEFAULT' or csv_content == 'HOMOGENEOUS':
            # a primeira coluna será DATETIME e a segunda em diante será B0, B1, B2 etc.
            self.df.columns = ['DATETIME'] + [f'B{i}' for i in range(0, len(self.df.columns) - 1)]
        else:
            # informe que o parâmetro csv_content é inválido e aborte
            print(f'erro. csv_content inválido ({csv_content})')
            exit(-1)

    def create_df_from_rates_(self) -> pd.DataFrame:
        """
        Cria um DataFrame a partir de uma lista de dicionários contendo os dados para formar as linhas e colunas da
        planilha.
        :return:
        """
        _df: pd.DataFrame
        _list_ = []
        for rate in self.rates:
            _row = [rate['DATETIME'], rate['OPEN'], rate['HIGH'], rate['LOW'], rate['CLOSE'], rate['TICKVOL']]
            _list_.append(_row)
        _df = pd.DataFrame(_list_)
        _df.columns = ['DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL']
        return _df

    def create_df_from_rates(self) -> pd.DataFrame:
        """
        Cria um DataFrame a partir de uma lista de dicionários contendo os dados para formar as linhas e colunas da
        planilha. O nome das colunas será o nome das chaves dos dicionários.
        :return:
        """
        _df: pd.DataFrame
        _list_ = []
        for rate in self.rates:
            _row = [i for i in rate.values()]
            _list_.append(_row)
        _df = pd.DataFrame(_list_)
        _df.columns = [i for i in self.rates[0]]
        return _df

    def get_timedelta(self) -> timedelta:
        tf = self.timeframe
        ret: timedelta

        if tf == 'M1':
            ret = timedelta(minutes=1)
        elif tf == 'M2':
            ret = timedelta(minutes=2)
        elif tf == 'M3':
            ret = timedelta(minutes=3)
        elif tf == 'M4':
            ret = timedelta(minutes=4)
        elif tf == 'M5':
            ret = timedelta(minutes=5)
        elif tf == 'M6':
            ret = timedelta(minutes=6)
        elif tf == 'M10':
            ret = timedelta(minutes=10)
        elif tf == 'M12':
            ret = timedelta(minutes=12)
        elif tf == 'M15':
            ret = timedelta(minutes=15)
        elif tf == 'M20':
            ret = timedelta(minutes=20)
        elif tf == 'M30':
            ret = timedelta(minutes=30)
        elif tf == 'H1':
            ret = timedelta(hours=1)
        elif tf == 'H2':
            ret = timedelta(hours=2)
        elif tf == 'H3':
            ret = timedelta(hours=3)
        elif tf == 'H4':
            ret = timedelta(hours=4)
        elif tf == 'H6':
            ret = timedelta(hours=6)
        elif tf == 'H8':
            ret = timedelta(hours=8)
        elif tf == 'H12':
            ret = timedelta(hours=12)
        elif tf == 'D1':
            ret = timedelta(days=1)
        elif tf == 'W1':
            ret = timedelta(weeks=1)
        else:
            print('erro. get_timedelta. timeframe inválido.')
            exit(-1)

        return ret

    def go_to_next_row(self):
        if self.current_row == len(self.df) - 1:
            return False
        else:
            self.current_row += 1
            if self.current_row == len(self.df) - 1:
                self.is_on_the_last_row = True
            return True

    def go_to_next_day(self):
        _datetime_prev = self.df.iloc[0]['DATETIME']
        df: DataFrame = self.df
        self.current_row += 1

        while True:
            row = df.iloc[self.current_row]
            _datetime = row['DATETIME']

            if _datetime != _datetime_prev:
                break

            _datetime_prev = _datetime
            self.current_row += 1

    def print_current_row(self):
        row = self.df.iloc[self.current_row]
        _datetime = row['DATETIME']

        # imprima o conteúdo da linha atual.
        # o que será impresso depende do conteúdo do arquivo CSV.
        # se for HETEROGENEOUS_OHLCV, imprima DATETIME, OPEN, HIGH, LOW, CLOSE, TICK
        # se for HETEROGENEOUS_DEFAULT, imprima DATETIME, B0, B1, B2 etc.
        # se for HOMOGENEOUS, imprima DATETIME, B0, B1, B2 etc.
        if self.csv_content == 'HETEROGENEOUS_OHLCV':
            _O, _H, _L, _C, _V = row['OPEN'], row['HIGH'], row['LOW'], row['CLOSE'], row['TICK_VOL']
            print(f'{self.symbol} {_datetime} (linha = {self.current_row}) '
                  f'OHLCV = {_O} {_H} {_L} {_C} {_V}')
        elif self.csv_content == 'HETEROGENEOUS_DEFAULT' or self.csv_content == 'HOMOGENEOUS':
            _B = [row[f'B{i}'] for i in range(0, len(row) - 1)]
            print(f'{self.symbol} {_datetime} (linha = {self.current_row}) B = {_B}')
        else:
            print(f'erro. csv_content inválido ({self.csv_content})')
            exit(-1)

    def print_last_row(self):
        if self.csv_content == 'HETEROGENEOUS_OHLCV':
            print(f'a última linha de {self.symbol}. ({len(self.df)} linhas)')
            row = self.df.iloc[-1]
            _datetime = row['DATETIME']
            _O, _H, _L, _C, _V = row['OPEN'], row['HIGH'], row['LOW'], row['CLOSE'], row['TICKVOL']
            print(f'{self.symbol} {_datetime} OHLCV = {_O} {_H} {_L} {_C} {_V}')
        elif self.csv_content == 'HETEROGENEOUS_DEFAULT' or self.csv_content == 'HOMOGENEOUS':
            print(f'a última linha de {self.symbol}. ({len(self.df)} linhas)')
            row = self.df.iloc[-1]
            _datetime = row['DATETIME']
            _B = [row[f'B{i}'] for i in range(0, len(row) - 1)]
            print(f'{self.symbol} {_datetime} B = {_B}')
        else:
            print(f'erro. csv_content inválido ({self.csv_content})')
            exit(-1)

    def get_datetime_last_row(self) -> datetime:
        row = self.df.iloc[-1]
        _datetime_str = row['DATETIME']
        _datetime = datetime.fromisoformat(_datetime_str)
        return _datetime


class SheetFile:
    def __init__(self, filepath: str, symbol: str, timeframe: str):
        print(f'criando planilha a partir de {filepath}')
        self.filepath = filepath
        self.symbol = symbol
        self.timeframe = timeframe
        self.timedelta: timedelta = self.get_timedelta()
        self.current_row = 0
        self.df: DataFrame = pd.read_csv(self.filepath, delimiter='\t')
        self.is_on_the_last_row = False

        if self.df.isnull().sum().values.sum() != 0:
            print(f'Há dados faltando no arquivo {self.filepath}')
            exit(-1)

    def get_timedelta(self) -> timedelta:
        tf = self.timeframe
        ret: timedelta

        if tf == 'M1':
            ret = timedelta(minutes=1)
        elif tf == 'M2':
            ret = timedelta(minutes=2)
        elif tf == 'M3':
            ret = timedelta(minutes=3)
        elif tf == 'M4':
            ret = timedelta(minutes=4)
        elif tf == 'M5':
            ret = timedelta(minutes=5)
        elif tf == 'M6':
            ret = timedelta(minutes=6)
        elif tf == 'M10':
            ret = timedelta(minutes=10)
        elif tf == 'M12':
            ret = timedelta(minutes=12)
        elif tf == 'M15':
            ret = timedelta(minutes=15)
        elif tf == 'M20':
            ret = timedelta(minutes=20)
        elif tf == 'M30':
            ret = timedelta(minutes=30)
        elif tf == 'H1':
            ret = timedelta(hours=1)
        elif tf == 'H2':
            ret = timedelta(hours=2)
        elif tf == 'H3':
            ret = timedelta(hours=3)
        elif tf == 'H4':
            ret = timedelta(hours=4)
        elif tf == 'H6':
            ret = timedelta(hours=6)
        elif tf == 'H8':
            ret = timedelta(hours=8)
        elif tf == 'H12':
            ret = timedelta(hours=12)
        elif tf == 'D1':
            ret = timedelta(days=1)
        elif tf == 'W1':
            ret = timedelta(weeks=1)
        else:
            print('erro. get_timedelta. timeframe inválido.')
            exit(-1)

        return ret

    def go_to_next_row(self):
        if self.current_row == len(self.df) - 1:
            return False
        else:
            self.current_row += 1
            if self.current_row == len(self.df) - 1:
                self.is_on_the_last_row = True
            return True

    def go_to_next_day(self):
        _datetime_prev = self.df.iloc[0]['DATETIME']
        df: DataFrame = self.df
        self.current_row += 1

        while True:
            row = df.iloc[self.current_row]
            _datetime = row['DATETIME']

            if _datetime != _datetime_prev:
                break

            _datetime_prev = _datetime
            self.current_row += 1

    def print_current_row(self):
        row = self.df.iloc[self.current_row]
        _datetime = row['DATETIME']
        _O, _H, _L, _C, _V = row['OPEN'], row['HIGH'], row['LOW'], row['CLOSE'], row['TICKVOL']
        print(f'{self.symbol} {_datetime} (linha = {self.current_row}) '
              f'OHLCV = {_O} {_H} {_L} {_C} {_V}')

    def print_last_row(self):
        print(f'a última linha de {self.symbol}. ({len(self.df)} linhas)')
        row = self.df.iloc[-1]
        _datetime = row['DATETIME']
        _O, _H, _L, _C, _V = row['OPEN'], row['HIGH'], row['LOW'], row['CLOSE'], row['TICKVOL']
        print(f'{self.symbol} {_datetime} OHLCV = {_O} {_H} {_L} {_C} {_V}')

    def get_datetime_last_row(self) -> datetime:
        row = self.df.iloc[-1]
        _datetime_str = row['DATETIME']
        _datetime = datetime.fromisoformat(_datetime_str)
        return _datetime


class SheetRates:
    def __init__(self, rates: list[dict], symbol: str, timeframe: str):
        print(f'criando planilha para {symbol}_{timeframe}')
        self.rates = rates
        self.symbol = symbol
        self.timeframe = timeframe
        self.timedelta: timedelta = self.get_timedelta()
        self.current_row = 0
        self.previous_close = 0.0
        self.df: DataFrame = self.create_df_from_rates()
        self.is_on_the_last_row = False
        self.is_trading = False

        if self.df.isnull().sum().values.sum() != 0:
            print(f'Há dados faltando no dataframe {symbol}_{timeframe}')
            exit(-1)

    def create_df_from_rates(self) -> pd.DataFrame:
        _df: pd.DataFrame
        _list_ = []
        for rate in self.rates:
            _row = [rate['DATETIME'], rate['OPEN'], rate['HIGH'], rate['LOW'], rate['CLOSE'], rate['TICKVOL']]
            _list_.append(_row)
        _df = pd.DataFrame(_list_)
        _df.columns = ['DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL']
        return _df

    def get_timedelta(self) -> timedelta:
        tf = self.timeframe
        ret: timedelta

        if tf == 'M1':
            ret = timedelta(minutes=1)
        elif tf == 'M2':
            ret = timedelta(minutes=2)
        elif tf == 'M3':
            ret = timedelta(minutes=3)
        elif tf == 'M4':
            ret = timedelta(minutes=4)
        elif tf == 'M5':
            ret = timedelta(minutes=5)
        elif tf == 'M6':
            ret = timedelta(minutes=6)
        elif tf == 'M10':
            ret = timedelta(minutes=10)
        elif tf == 'M12':
            ret = timedelta(minutes=12)
        elif tf == 'M15':
            ret = timedelta(minutes=15)
        elif tf == 'M20':
            ret = timedelta(minutes=20)
        elif tf == 'M30':
            ret = timedelta(minutes=30)
        elif tf == 'H1':
            ret = timedelta(hours=1)
        elif tf == 'H2':
            ret = timedelta(hours=2)
        elif tf == 'H3':
            ret = timedelta(hours=3)
        elif tf == 'H4':
            ret = timedelta(hours=4)
        elif tf == 'H6':
            ret = timedelta(hours=6)
        elif tf == 'H8':
            ret = timedelta(hours=8)
        elif tf == 'H12':
            ret = timedelta(hours=12)
        elif tf == 'D1':
            ret = timedelta(days=1)
        elif tf == 'W1':
            ret = timedelta(weeks=1)
        else:
            print('erro. get_timedelta. timeframe inválido.')
            exit(-1)

        return ret


if __name__ == '__main__':
    _filepath = '../csv/EURUSD_M5.csv'
    s = Sheet(_filepath, 'EURUSD', 'M5')
    s.print_last_row()
