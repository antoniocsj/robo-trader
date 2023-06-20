import os
import math
import pickle
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from pandas import DataFrame
import multiprocessing as mp


def search_symbols_in_directory(directory: str, timeframe: str) -> list[str]:
    """
    Procurando pelos símbolos presentes num diretório contendo arquivos csv.
    Todos os arquivos devem ser do mesmo timeframe
    :return: lista dos símbolos
    """
    # passe por todos os arquivos csv e descubra o symbol e timeframe
    symbols = []
    all_files = os.listdir(directory)

    for filename in all_files:
        if filename.endswith('.csv'):
            _symbol = filename.split('_')[0]
            _timeframe = filename.split('_')[1]
            if _timeframe.endswith('.csv'):
                _timeframe = _timeframe.replace('.csv', '')

            if _symbol not in symbols:
                symbols.append(_symbol)
            else:
                print(f'erro. o símbolo {_symbol} aparece repetido no mesmo diretório')
                exit(-1)

            if _timeframe != timeframe:
                print(f'erro. timeframe {_timeframe} diferente do especificado {timeframe} foi '
                      f'encontrado no diretório')
                exit(-1)

    symbols = sorted(symbols)
    return symbols


class Sheet:
    def __init__(self, filepath: str, symbol: str, timeframe: str):
        print(f'criando planilha a partir de {filepath}')
        self.filepath = filepath
        self.symbol = symbol
        self.timeframe = timeframe
        self.timedelta: timedelta = self.get_timedelta()
        self.current_row = 0
        self.previous_close = 0.0
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
            self.previous_close = self.df.iloc[self.current_row - 1]['<CLOSE>']
            if self.current_row == len(self.df) - 1:
                self.is_on_the_last_row = True
            return True

    def go_to_next_day(self):
        _date_prev = self.df.iloc[0]['<DATE>']
        df: DataFrame = self.df
        self.current_row += 1

        while True:
            row = df.iloc[self.current_row]
            _date = row['<DATE>']

            if _date != _date_prev:
                break

            _date_prev = _date
            self.current_row += 1
            self.previous_close = self.df.iloc[self.current_row - 1]['<CLOSE>']

    def print_current_row(self):
        row = self.df.iloc[self.current_row]
        _date, _time = row['<DATE>'], row['<TIME>']
        _O, _H, _L, _C, _V = row['<OPEN>'], row['<HIGH>'], row['<LOW>'], row['<CLOSE>'], row['<TICKVOL>']
        print(f'{self.symbol} {_date} {_time} (linha = {self.current_row}) '
              f'OHLCV = {_O} {_H} {_L} {_C} {_V}')

    def print_last_row(self):
        print(f'a última linha de {self.symbol}. ({len(self.df)} linhas)')
        row = self.df.iloc[-1]
        _date, _time = row['<DATE>'], row['<TIME>']
        _O, _H, _L, _C, _V = row['<OPEN>'], row['<HIGH>'], row['<LOW>'], row['<CLOSE>'], row['<TICKVOL>']
        print(f'{self.symbol} {_date} {_time} OHLCV = {_O} {_H} {_L} {_C} {_V}')

    def get_datetime_last_row(self) -> datetime:
        row = self.df.iloc[-1]
        _date = row['<DATE>'].replace('.', '-')
        _time = row['<TIME>']
        _date_time_str = f"{_date}T{_time}"
        _date_time = datetime.fromisoformat(_date_time_str)

        return _date_time


class DirectoryCorrection:
    def __init__(self, directory: str, timeframe: str, index_proc: int, symbols_to_sync: list[str] = None):
        self.directory = directory
        self.all_files = []
        self.csv_files = {}
        self.symbols = []
        self.timeframe = timeframe
        self.sheets = []
        self.num_insertions_done = 0
        self.exclude_first_rows = False
        self.new_start_row_datetime: datetime = None
        self.cp = {}
        self.index_proc = index_proc

        self.search_symbols(symbols_to_sync)
        self.load_sheets()
        self.create_checkpoint(index_proc)

    def search_symbols(self, symbols_to_sync: list[str] = None):
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
                    if not symbols_to_sync:
                        self.symbols.append(_symbol)
                    else:
                        if _symbol in symbols_to_sync:
                            self.symbols.append(_symbol)
                        else:
                            continue
                else:
                    print(f'erro. o símbolo {_symbol} aparece repetido no mesmo diretório')
                    exit(-1)

                if _timeframe != self.timeframe:
                    print(f'erro. {_timeframe} timeframe inválido')
                    exit(-1)

                self.csv_files[f'{_symbol}_{_timeframe}'] = filename

        self.symbols = sorted(self.symbols)

    def get_csv_filepath(self, _symbol_timeframe: str) -> str:
        _filepath = self.directory + '/' + self.csv_files[_symbol_timeframe]
        return _filepath

    def load_sheets(self):
        for _symbol in self.symbols:
            _symbol_timeframe = f'{_symbol}_{self.timeframe}'
            _filepath = self.get_csv_filepath(_symbol_timeframe)
            self.sheets.append(Sheet(_filepath, _symbol, self.timeframe))

    def _calc_max(self, _list: list[tuple[datetime, Sheet]]) -> tuple[datetime, Sheet]:
        _max_datetime_sheet = (datetime.min, None)
        for e in _list:
            if e[0] > _max_datetime_sheet[0]:
                _max_datetime_sheet = e
        return _max_datetime_sheet

    def _calc_min(self, _list: list[tuple[datetime, Sheet]]) -> tuple[datetime, Sheet]:
        _min_datetime_sheet = (datetime.max, None)
        for e in _list:
            if e[0] < _min_datetime_sheet[0]:
                _min_datetime_sheet = e
        return _min_datetime_sheet

    def find_first_row(self):
        print(f'find_first_row() index_proc = {self.index_proc}')

        #  analisando a primeira linha de cada planilha.
        s: Sheet
        _datetime_sheet_list = []
        _date_time_list = []
        for s in self.sheets:
            row = s.df.iloc[0]
            _date = row['<DATE>'].replace('.', '-')
            _time = row['<TIME>']
            _date_time_str = f"{_date}T{_time}"
            _date_time = datetime.fromisoformat(_date_time_str)
            _datetime_sheet_list.append((_date_time, s))

        _date_time_list = [t[0] for t in _datetime_sheet_list]
        _date_time_set = set(_date_time_list)

        if len(_date_time_set) == 1:
            # todas as planilhas estão sincronizadas nesta linha.
            # se todas as planilhas começam na mesma data e horário, então apenas retorne.
            # pois todas as planilhas já tem o current_row ajustado para 0 inicialmente.
            print('todas as planilhas começam na mesma data e horário.')
            for s in self.sheets:
                s.print_current_row()
            print()
            return

        print('nem todas as planilhas começam na mesma data e horário.')
        print(f'datas/horários encontrados na primeira linha de todas as planilhas: {_date_time_set}')

        # buscando uma nova linha que será o novo começo.
        # para que uma data ou horário sejam válidos, tem que estar presente em todas as planilhas.
        # _highest_datetime = max(_date_time_set)
        # _highest_datetime_sheet = max(_datetime_sheet_list)
        _highest_datetime_sheet = self._calc_max(_datetime_sheet_list)

        # continua aqui
        while True:
            _candidates_list = []
            _counter = 0
            for s in self.sheets:
                i = 0
                while True:
                    row = s.df.iloc[i]
                    _date = row['<DATE>'].replace('.', '-')
                    _time = row['<TIME>']
                    _date_time_str = f"{_date}T{_time}"
                    _date_time = datetime.fromisoformat(_date_time_str)
                    if _date_time == _highest_datetime_sheet[0]:
                        _counter += 1
                        _candidates_list.append((_date_time, s))
                        break
                    elif _date_time > _highest_datetime_sheet[0]:
                        _candidates_list.append((_date_time, s))
                        break
                    i += 1

            _list = [t[0] for t in _candidates_list]
            print(f'candidatos: {_list}')
            _highest_datetime_sheet = self._calc_max(_candidates_list)
            if _counter == len(self.sheets):
                # encontramos uma linha sincronizada em todas as planilhas
                print(f'novo início em {_highest_datetime_sheet[0]}')
                break

        # excluir as linhas iniciais de todas as planilhas.
        for s in self.sheets:
            i = 0
            while True:
                row = s.df.iloc[i]
                _date = row['<DATE>'].replace('.', '-')
                _time = row['<TIME>']
                _date_time_str = f"{_date}T{_time}"
                _date_time = datetime.fromisoformat(_date_time_str)
                if _date_time == _highest_datetime_sheet[0]:
                    s.df.drop(s.df.index[0:i], inplace=True)
                    s.df.sort_index(ignore_index=True, inplace=True)
                    s.df.reset_index(drop=True)
                    _filepath = self.get_csv_filepath(f'{s.symbol}_{s.timeframe}')
                    print(f'salvando arquivo {_filepath}')
                    s.df.to_csv(_filepath, sep='\t', index=False)
                    break
                i += 1

    def correct_directory(self, index_proc: int):
        print(f'\n*** process index : {index_proc}')
        print(f'*** symbols to sync : {self.symbols}\n')

        _r, _current_row, finished = self.open_checkpoint(index_proc)
        if _r:
            print(f'checkpoint {index_proc} carregado. linha atual = {_current_row}')

            if finished:
                print(f'checkpoint {index_proc} indica que símbolos já estão sincronizados.')
                return
            else:
                self.sheets_set_current_row(_current_row)

        self.find_first_row()

        _counter = 0
        _max_len = 0
        _len_list = []

        # descobre qual planilha tem o maior número de linhas
        for s in self.sheets:
            _len_list.append(len(s.df))
        _max_len = max(_len_list)

        # percorre todas as linhas de todas as planilhas.
        _sheet_reached_the_end: Sheet = None
        while True:
            s: Sheet

            if _sheet_reached_the_end:
                print('a sincronização está concluída')
                break

            self.insert_rows()

            for s in self.sheets:
                _r = s.go_to_next_row()
                _current_row = s.current_row
                if _r is False:
                    _sheet_reached_the_end = s
                    self.sheets_exclude_last_rows(s.current_row)
                    break

            if _current_row % 10000 == 0 and _current_row > 0:
                self.save_sheets(print_row='current')
                self.write_checkpoint(index_proc)
                print(f'{100 * _current_row / _max_len: .2f} %\n')

            # if _current_row % 100 == 0 and _current_row > 0:
            #     print('\nrelatório parcial.')
            #     for s in self.sheets:
            #         s.print_current_row()
            #     if self.all_sheets_datetime_synced_this_row():
            #         print(f'OK! TODAS as planilhas estão SINCRONIZADAS até aqui. (linha {_current_row})')
            #     else:
            #         print('ERRO! NEM todas as planilhas estão sincronizadas até aqui.')
            #     print()

        if self.check_sheets_last_row():
            self.save_sheets()

        print()
        for s in self.sheets:
            s.print_current_row()
        if self.all_sheets_datetime_synced_this_row():
            print('OK! TODAS as planilhas estão SINCRONIZADAS até aqui.')
            self.write_checkpoint(index_proc, finished=True)
        else:
            print('ERRO! NEM todas as planilhas estão sincronizadas até aqui.')
        print()

    def insert_rows(self):
        """
        Se for necessário, faz inserções de linhas nas planilhas.
        :return:
        """
        # verifique se a linha atual de cada planilha contém a mesma data e horário.
        s: Sheet
        _datetime_sheet_list = []
        _date_time_list = []
        for s in self.sheets:
            row = s.df.iloc[s.current_row]
            _date = row['<DATE>'].replace('.', '-')
            _time = row['<TIME>']
            _date_time_str = f"{_date}T{_time}"
            _date_time = datetime.fromisoformat(_date_time_str)
            _datetime_sheet_list.append((_date_time, s))

        _date_time_list = [t[0] for t in _datetime_sheet_list]
        _date_time_set = set(_date_time_list)
        if len(_date_time_set) == 1:
            # todas as planilhas estão sincronizadas nesta linha.
            return

        # nem todas as planilhas estão sincronizadas nesta linha.
        # descubra qual planilha possui o menor 'datetime'.
        # _lower_datetime = min(_date_time_list)
        _lower_datetime_sheet = self._calc_min(_datetime_sheet_list)

        # o menor datetime será a referência. se alguma planilha tiver um datetime maior,
        # então essa planilha sofrerá inserções de novas linhas até que esteja sincronizada.
        for s in self.sheets:
            _row = s.df.iloc[s.current_row]
            _date = _row['<DATE>'].replace('.', '-')
            _time = _row['<TIME>']
            _date_time_str = f"{_date}T{_time}"
            _date_time = datetime.fromisoformat(_date_time_str)

            if _date_time > _lower_datetime_sheet[0]:
                _lower_datetime, _lower_sheet = _lower_datetime_sheet[0], _lower_datetime_sheet[1]
                print(f'{s.symbol} {_date_time} > {_lower_datetime} ({_lower_sheet.symbol}) '
                      f'(linha atual = {s.current_row}) (index_proc = {self.index_proc})')
                _previous_row = s.df.iloc[s.current_row - 1]
                _previous_close = _previous_row['<CLOSE>']

                # faz as inserções de novas linhas até _date_time. os datetime's das linhas inseridas
                # começam em _lower_datetime e vão até (mas não incluindo) _date_time.
                _new_date_time = _lower_datetime
                _index_start = s.current_row
                _index_new_row = s.current_row - 0.5

                while _new_date_time < _date_time:
                    is_present = self.is_present_inother_sheets(_new_date_time,
                                                                _lower_datetime_sheet[1].current_row,
                                                                s.current_row)
                    if not is_present:
                        _new_date_time += s.timedelta
                        continue

                    _index_new_row = self.insert_new_row(_index_new_row, _new_date_time,
                                                         _previous_close, s)

                    _new_date_time += s.timedelta

                self.num_insertions_done += 1
                s.current_row = _index_start

    def insert_new_row(self, _index_new_row, _new_date_time, _previous_close, s):
        # print(f'{s.symbol} inserindo nova linha {_new_date_time}')
        _date = _new_date_time.strftime('%Y.%m.%d')
        _time = _new_date_time.strftime('%H:%M')
        _O = _H = _L = _C = _previous_close
        # insere a nova linha. que será uma vela com O=H=L=C igual a _previous_close e V=0
        # s.df.loc[_index_new_row] = [_date, _time, _O, _H, _L, _C, 0, 0, 0]
        s.df.loc[_index_new_row] = [_date, _time, _O, _H, _L, _C, 0]
        s.df.sort_index(ignore_index=True, inplace=True)
        s.df.reset_index(drop=True)
        s.current_row += 1
        _index_new_row = s.current_row - 0.5
        return _index_new_row

    def is_present_inother_sheets(self, _new_date_time, _nrow_start, _nrow_end):
        s: Sheet
        _date_time_list = []
        for s in self.sheets:
            for i in range(_nrow_start, _nrow_end + 1):
                if i > len(s.df) - 1:
                    break
                row = s.df.iloc[i]
                _date = row['<DATE>'].replace('.', '-')
                _time = row['<TIME>']
                _date_time_str = f"{_date}T{_time}"
                _date_time = datetime.fromisoformat(_date_time_str)
                _date_time_list.append(_date_time)

        _ret = _new_date_time in _date_time_list
        return _ret

    def sheets_exclude_last_rows(self, current_row):
        for s in self.sheets:
            i = current_row + 1
            if i > len(s.df) - 1:
                continue
            s.df.drop(s.df.index[i:], inplace=True)
            s.df.sort_index(ignore_index=True, inplace=True)
            s.df.reset_index(drop=True)
            s.current_row = s.df.index[-1]

    def check_sheets_last_row(self) -> bool:
        _date_time_set = set()
        _len_set = set()
        s: Sheet
        for s in self.sheets:
            _date_time_set.add(s.get_datetime_last_row())
            _len_set.add(len(s.df))
        if len(_date_time_set) == 1 and len(_len_set) == 1:
            print('as últimas linhas estão sincronizadas')
            return True
        else:
            print('as últimas linhas NÃO estão sincronizadas')
            return False

    def save_sheets(self, print_row='last'):
        s: Sheet
        for s in self.sheets:
            _filepath = self.get_csv_filepath(f'{s.symbol}_{s.timeframe}')
            print(f'salvando arquivo {_filepath}. linha atual = {s.current_row}')

            if print_row == 'last':
                s.print_last_row()
            elif print_row == 'current':
                s.print_current_row()

            s.df.to_csv(_filepath, sep='\t', index=False)

    def sheets_set_current_row(self, _current_row):
        s: Sheet
        for s in self.sheets:
            s.current_row = _current_row

    def open_checkpoint(self, index_proc: int):
        _filename = f'sync_cp_{index_proc}.pkl'
        if os.path.exists(_filename):
            with open(_filename, 'rb') as file:
                self.cp = pickle.load(file)
            current_row = self.cp['current_row']
            finished = self.cp['finished']
            return True, current_row, finished
        return False, 0, False

    # def write_checkpoint(self, index_proc: int):
    #     s: Sheet
    #     _rows_set = set()
    #     for s in self.sheets:
    #         _rows_set.add(s.current_row)
    #     if self.all_sheets_row_synced():
    #         _current_row = list(_rows_set)[0]
    #         self.cp['current_row'] = _current_row
    #         with open(f'sync_cp_{index_proc}.pkl', 'wb') as file:
    #             pickle.dump(self.cp, file)
    #         print(f'checkpoint gravado. linha atual = {_current_row}\n')
    #     else:
    #         print(f'erro. write_checkpoint. as planilhas NÃO estão sincronizadas. '
    #               f'current_rows = {list(_rows_set)}\n')
    #         exit(-1)

    def write_checkpoint(self, index_proc: int, finished=False):
        s: Sheet
        _rows_set = set()
        for s in self.sheets:
            _rows_set.add(s.current_row)
        if not self.all_sheets_row_synced():
            print(f'cuidado. write_checkpoint. as planilhas NÃO estão todas sincronizadas ainda. '
                  f'current_rows = {list(_rows_set)}\n')

        _current_row = list(_rows_set)[0]
        self.cp['symbols_to_sync'] = self.symbols
        self.cp['finished'] = finished
        self.cp['current_row'] = _current_row

        _filename = f'sync_cp_{index_proc}.pkl'
        with open(_filename, 'wb') as file:
            pickle.dump(self.cp, file)
        print(f'checkpoint {_filename} gravado. linha atual = {_current_row}\n')

    def create_checkpoint(self, index_proc: int):
        _filename = f'sync_cp_{index_proc}.pkl'
        if os.path.exists(_filename):
            return

        self.cp['symbols_to_sync'] = self.symbols
        self.cp['finished'] = False
        self.cp['current_row'] = 0

        _filename = f'sync_cp_{index_proc}.pkl'
        with open(_filename, 'wb') as file:
            pickle.dump(self.cp, file)
        print(f'checkpoint {_filename} criado.\n')

    def all_sheets_datetime_synced_this_row(self):
        """
        Testa se todas as planilhas estão sincronizadas (mesma data e hora) nesta linha.
        :return: True, se estiverem sinzronizadas, False, caso contrário.
        """
        s: Sheet
        _rows_set = set()
        _datetime_set = set()

        for s in self.sheets:
            row = s.df.iloc[s.current_row]
            _date = row['<DATE>'].replace('.', '-')
            _time = row['<TIME>']
            _datetime_str = f"{_date}T{_time}"
            _rows_set.add(s.current_row)
            _datetime_set.add(_datetime_str)

        if len(_rows_set) == 1 and len(_datetime_set) == 1:
            return True
        else:
            return False

    def all_sheets_row_synced(self):
        """
        Testa se todas as planilhas estão sincronizadas (mesmo current_row) nesta linha.
        :return: True, se estiverem sinzronizadas, False, caso contrário.
        """
        s: Sheet
        _rows_set = set()

        for s in self.sheets:
            _rows_set.add(s.current_row)

        if len(_rows_set) == 1:
            return True
        else:
            return False

    def check(self):
        """
        Verifica se todas as linhas estão com a data/hora em ordem crescente.
        Se houver alguma linha que desobedece essa regra, reporta o evento.
        :return:
        """
        s: Sheet
        for s in self.sheets:
            print(s.symbol)
            row = s.df.iloc[0]
            _date = row['<DATE>'].replace('.', '-')
            _time = row['<TIME>']
            _datetime_str = f"{_date}T{_time}"
            _datetime_previous = datetime.fromisoformat(_datetime_str)
            for i in range(1, len(s.df)):
                row = s.df.iloc[i]
                _date = row['<DATE>'].replace('.', '-')
                _time = row['<TIME>']
                _datetime_str = f"{_date}T{_time}"
                _datetime_current = datetime.fromisoformat(_datetime_str)
                if _datetime_current <= _datetime_previous:
                    print(f'erro em {s.symbol} {_datetime_current}')
                _datetime_previous = _datetime_current


def get_list_sync_files():
    _list = []
    all_files = os.listdir('.')

    for filename in all_files:
        if filename.startswith('sync_cp_') and filename.endswith('.pkl'):
            _list.append(filename)

    return sorted(_list)


def get_sync_status(_filename: str):
    if os.path.exists(_filename):
        with open(_filename, 'rb') as file:
            cp = pickle.load(file)
        finished = cp['finished']
        return finished
    return False


def get_symbols_to_sync(_filename: str) -> list[str] | None:
    if os.path.exists(_filename):
        with open(_filename, 'rb') as file:
            cp = pickle.load(file)
        _symbols = cp['symbols_to_sync']
        return _symbols
    return None


def get_all_sync_cp_dic(_list_sync_files: list[str]) -> list[dict]:
    _list = []
    for _filename in _list_sync_files:
        if os.path.exists(_filename):
            with open(_filename, 'rb') as file:
                cp = pickle.load(file)
                _list.append(cp)
        else:
            print(f'erro em get_all_sync_cp_dic(). arquivo {_filename} não foi encontrado.')
            exit(-1)

    return _list


def create_sync_cp_file(index_proc: int, _symbols_to_sync: list[str]):
    _filename = f'sync_cp_{index_proc}.pkl'
    _cp = {'symbols_to_sync': _symbols_to_sync,
           'finished': False,
           'current_row': 0}

    _filename = f'sync_cp_{index_proc}.pkl'
    with open(_filename, 'wb') as file:
        pickle.dump(_cp, file)
    print(f'checkpoint {_filename} criado.')


def remove_sync_cp_files(_list_sync_files: list[str]):
    for filename in _list_sync_files:
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print(f'erro em remove_sync_cp_files(). arquivo {filename} não encontrado.')
            exit(-1)


def main():
    dir_csv = '../csv'
    timeframe = 'M10'
    symbols = search_symbols_in_directory(dir_csv, timeframe)
    symbols_to_sync_per_proc = []

    list_sync_files = get_list_sync_files()
    if len(list_sync_files) == 0:
        print('iniciando a sincronização dos arquivos csv pela PRIMEIRA vez.')
        n_procs = 8
        pool = mp.Pool(n_procs)

        for i in range(n_procs):
            symbols_to_sync_per_proc.append([])

        for i in range(len(symbols)):
            i_proc = i % n_procs
            symbols_to_sync_per_proc[i_proc].append(symbols[i])

        dir_cor_l: list[DirectoryCorrection] = []
        for i in range(n_procs):
            dir_cor = DirectoryCorrection(dir_csv, timeframe, i, symbols_to_sync_per_proc[i])
            dir_cor_l.append(dir_cor)

        for i in range(n_procs):
            pool.apply_async(dir_cor_l[i].correct_directory, args=(i, ))
        pool.close()
        pool.join()
        print('*************************************')

    else:
        print('pode haver sincronização em andamento.')
        print(f'checkpoints: {list_sync_files}')
        # verifique se todos os checkpoints indicam sincronização finalizada
        _results_set = set()
        for i in range(len(list_sync_files)):
            _sync_status = get_sync_status(list_sync_files[i])
            _results_set.add(_sync_status)
        if len(_results_set) == 1 and list(_results_set)[0] is True:
            print('TODOS os checkpoints indicam que suas sincronização estão FINALIZADAS')
            # assume-se que sempre há um número de processos/num_sync_cp_files que seja uma potência de 2
            # pois será feita uma fusão de cada 2 conjuntos de símbolos de modo que na próxima sincronização
            # haverá a metade do número de processos/num_sync_cp_files do que havia na sincronização precedente.
            n_procs = len(list_sync_files)
            if n_procs == 1:
                print('a sincronização total está finalizada. parabéns!')
            else:
                print(f'iniciando a fusão de conjuntos de símbolos')
                list_sync_cp_dic = get_all_sync_cp_dic(list_sync_files)
                n_procs = n_procs // 2
                pool = mp.Pool(n_procs)
                symbols_to_sync_per_proc = []

                for i in range(n_procs):
                    symbols_to_sync_per_proc.append([])

                for i in range(n_procs * 2):
                    j = math.floor(i / 2)
                    symbols_to_sync_per_proc[j] += list_sync_cp_dic[i]['symbols_to_sync']

                print('removendo sync_cp_files')
                remove_sync_cp_files(get_list_sync_files())

                print('criando novo(s) sync_cp_file(s)')
                for i in range(n_procs):
                    create_sync_cp_file(i, symbols_to_sync_per_proc[i])

                dir_cor_l: list[DirectoryCorrection] = []
                for i in range(n_procs):
                    dir_cor = DirectoryCorrection(dir_csv, timeframe, i, symbols_to_sync_per_proc[i])
                    dir_cor_l.append(dir_cor)

                for i in range(n_procs):
                    pool.apply_async(dir_cor_l[i].correct_directory, args=(i,))
                pool.close()
                pool.join()
                print('*************************************')
        else:
            print('NEM todos os checkpoints indicam que suas sincronização estão finalizadas')
            print('continuando as sincronizações')
            n_procs = len(list_sync_files)
            pool = mp.Pool(n_procs)

            for i in range(n_procs):
                symbols_to_sync_per_proc.append([])

            for i in range(n_procs):
                symbols_to_sync_per_proc[i] = get_symbols_to_sync(list_sync_files[i])

            dir_cor_l: list[DirectoryCorrection] = []
            for i in range(n_procs):
                dir_cor = DirectoryCorrection(dir_csv, timeframe, i, symbols_to_sync_per_proc[i])
                dir_cor_l.append(dir_cor)

            for i in range(n_procs):
                pool.apply_async(dir_cor_l[i].correct_directory, args=(i,))
            pool.close()
            pool.join()
            print('*************************************')


if __name__ == '__main__':
    main()
