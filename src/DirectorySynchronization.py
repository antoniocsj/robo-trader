import os
import json
from datetime import datetime
from utils_symbols import search_symbols_in_directory
from utils_filesystem import read_json, make_backup
from Sheet import Sheet


class DirectorySynchronization:
    """
    Realiza a correção/sincronização de todos os arquivos CSVs contidos no diretório indicado.
    Quando realiza inserções de linhas, faz do seguinte modo:
    - Obtém o valor de fechamento da última vela conhecida (C');
    - as velas inseridas terão O=H=L=C=C' e V=0;
    Todas as planilhas são sincronizadas simultaneamente em apenas 1 processo.
    """
    def __init__(self, directory: str):
        self.directory = directory
        self.all_files = []
        self.csv_files = {}
        self.symbols = []
        self.timeframe = ''
        self.sheets = []
        self.num_insertions_done = 0
        self.exclude_first_rows = False
        self.new_start_row_datetime: datetime = None
        self.cp = {}

        self.search_symbols()
        self.load_sheets()

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
        print('find_first_row')

        #  analisando a primeira linha de cada planilha.
        s: Sheet
        _datetime_sheet_list = []
        _datetime_list = []
        for s in self.sheets:
            row = s.df.iloc[0]
            _datetime_str = row['DATETIME']
            _datetime = datetime.fromisoformat(_datetime_str)
            _datetime_sheet_list.append((_datetime, s))

        _datetime_list = [t[0] for t in _datetime_sheet_list]
        _datetime_set = set(_datetime_list)

        if len(_datetime_set) == 1:
            # todas as planilhas estão sincronizadas nesta linha.
            # se todas as planilhas começam na mesma data e horário, então apenas retorne.
            # pois todas as planilhas já tem o current_row ajustado para 0 inicialmente.
            print('todas as planilhas começam na mesma data e horário.')
            for s in self.sheets:
                s.print_current_row()
            print()
            return

        print('nem todas as planilhas começam na mesma data e horário.')
        print(f'datas/horários encontrados na primeira linha de todas as planilhas: {_datetime_set}')

        # buscando uma nova linha que será o novo começo.
        # para que uma data ou horário sejam válidos, tem que estar presente em todas as planilhas.
        # _highest_datetime = max(_datetime_set)
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
                    _datetime_str = row['DATETIME']
                    _datetime = datetime.fromisoformat(_datetime_str)
                    if _datetime == _highest_datetime_sheet[0]:
                        _counter += 1
                        _candidates_list.append((_datetime, s))
                        break
                    elif _datetime > _highest_datetime_sheet[0]:
                        _candidates_list.append((_datetime, s))
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
                _datetime_str = row['DATETIME']
                _datetime = datetime.fromisoformat(_datetime_str)
                if _datetime == _highest_datetime_sheet[0]:
                    s.df.drop(s.df.index[0:i], inplace=True)
                    s.df.sort_index(ignore_index=True, inplace=True)
                    s.df.reset_index(drop=True)
                    _filepath = self.get_csv_filepath(f'{s.symbol}_{s.timeframe}')
                    print(f'salvando arquivo {_filepath}')
                    s.df.to_csv(_filepath, sep='\t', index=False)
                    break
                i += 1

    def synchronize_directory(self):
        setup = read_json('settings.json')
        temp_dir = setup['temp_dir']
        csv_s_dir = setup['csv_s_dir']

        _len_symbols = len(self.symbols)
        if _len_symbols == 0:
            print('Não há arquivos para sincronizar.')
            return
        elif _len_symbols == 1:
            print('Apenas 1 arquivo, portanto não há necessidade de sincronização.')
            return

        _r, _current_row, finished = self.open_checkpoint()
        if _r:
            print(f'checkpoint carregado. linha atual = {_current_row}')

            if finished:
                print(f'o checkpoint indica que os símbolos já estão sincronizados.')
                make_backup(temp_dir, csv_s_dir)
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
        while True:
            s: Sheet
            self.insert_rows()

            _list_s_r = []
            _set_r = set()
            for s in self.sheets:
                _r = s.go_to_next_row()
                _current_row = s.current_row
                _list_s_r.append((s, _r))
                _set_r.add(_r)

            # todas as planilhas avançaram para a próxima linha?
            # se todas NÃO avançaram, então chegamos ao final da sincronização;
            # se todas avançaram, basta continuar normalmente;
            # pode ocorrer de algumas avançarem e outras não, isso significa que algumas chegaram no
            # final precocemente. Nesse caso, as últimas deve receber novas linhas inseridas de modo a alcançarem
            # as outras que avançaram.
            if len(_set_r) == 1 and list(_set_r)[0] is False:
                print('fim da sincronização.')
                break
            elif len(_set_r) == 2:
                # crie 2 listas: uma de quem conseguiu avançar para a próxima linha e outra de quem não conseguiu.
                _ss_managed_to_advance = []
                _ss_failed_to_advance = []
                for s, _r in _list_s_r:
                    if _r is True:
                        _ss_managed_to_advance.append(s)
                    else:
                        _ss_failed_to_advance.append(s)

                # qual é a nova linha atual de quem conseguiu avançar para a próxima linha?
                _new_current_row = _ss_managed_to_advance[0].current_row

                # enquanto tiver planilhas que não conseguem avançar ao mesmo tempo que outras conseguem, então
                # não estamos no final verdadeiro. as planilhas que não estão conseguindo avançar para a próxima linha
                # estão tendo um final precoce. insira linhas no final delas até elas atingirem a mesma linha das
                # outras que avançaram.
                for s in _ss_failed_to_advance:
                    self.append_rows_until(s, _new_current_row)

            if _current_row % 20000 == 0 and _current_row > 0:
                self.save_sheets(print_row='current')
                self.write_checkpoint()
                print(f'{100 * _current_row / _max_len: .2f} %\n')

            # if _current_row % 100 == 0 and _current_row > 0:
            #     print('\nrelatório parcial.')
            #     for s in self.sheets:
            #         s.print_current_row()
            #     if self.all_sheets_datetime_synced_this_row():
            #         print('OK! TODAS as planilhas estão SINCRONIZADAS até aqui.')
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
            self.write_checkpoint(finished=True)
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
        _datetime_list = []
        for s in self.sheets:
            row = s.df.iloc[s.current_row]
            _datetime_str = row['DATETIME']
            _datetime = datetime.fromisoformat(_datetime_str)
            _datetime_sheet_list.append((_datetime, s))

        _datetime_list = [t[0] for t in _datetime_sheet_list]
        _datetime_set = set(_datetime_list)
        if len(_datetime_set) == 1:
            # todas as planilhas estão sincronizadas nesta linha.
            return

        # nem todas as planilhas estão sincronizadas nesta linha.
        # descubra qual planilha possui o menor 'datetime'.
        # _lower_datetime = min(_datetime_list)
        _lower_datetime_sheet = self._calc_min(_datetime_sheet_list)

        # o menor datetime será a referência. se alguma planilha tiver um datetime maior,
        # então essa planilha sofrerá inserções de novas linhas até que esteja sincronizada.
        for s in self.sheets:
            _row = s.df.iloc[s.current_row]
            _datetime_str = _row['DATETIME']
            _datetime = datetime.fromisoformat(_datetime_str)

            if _datetime > _lower_datetime_sheet[0]:
                _lower_datetime, _lower_sheet = _lower_datetime_sheet[0], _lower_datetime_sheet[1]
                print(f'{s.symbol} {_datetime} > {_lower_datetime} ({_lower_sheet.symbol}) '
                      f'(linha atual = {s.current_row})')
                _previous_row = s.df.iloc[s.current_row - 1]
                _previous_close = _previous_row['CLOSE']

                # faz as inserções de novas linhas até _datetime. os datetime's das linhas inseridas
                # começam em _lower_datetime e vão até (mas não incluindo) _datetime.
                _new_date_time = _lower_datetime
                _index_start = s.current_row
                _index_new_row = s.current_row - 0.5

                while _new_date_time < _datetime:
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

    def insert_new_row(self, _index_new_row, _new_datetime, _previous_close, s):
        """
        Insere um nova linha no dataframe. insere no meio, não no final. Para adicionar no final use append_new_row.
        :param _index_new_row:
        :param _new_datetime:
        :param _previous_close:
        :param s:
        :return:
        """
        # print(f'{s.symbol} inserindo nova linha {_new_date_time}')
        _datetime = _new_datetime.strftime('%Y-%m-%dT%H:%M')
        _O = _H = _L = _C = _previous_close
        # insere a nova linha. que será uma vela com O=H=L=C igual a _previous_close e V=0
        # s.df.loc[_index_new_row] = [_date, _time, _O, _H, _L, _C, 0, 0, 0]
        # s.df.loc[_index_new_row] = [_date, _time, _O, _H, _L, _C, 0]
        s.df.loc[_index_new_row] = [_datetime, _O, _H, _L, _C, 0]
        s.df.sort_index(ignore_index=True, inplace=True)
        s.df.reset_index(drop=True)
        s.current_row += 1
        _index_new_row = s.current_row - 0.5
        return _index_new_row

    def append_new_row(self, _index_new_row, _new_datetime, _previous_close, s):
        """
        Adiciona uma nova linha no final.
        :param _index_new_row:
        :param _new_datetime:
        :param _previous_close:
        :param s:
        :return:
        """
        # print(f'{s.symbol} inserindo nova linha {_new_date_time}')
        _datetime = _new_datetime.strftime('%Y-%m-%dT%H:%M')
        _O = _H = _L = _C = _previous_close
        # insere a nova linha. que será uma vela com O=H=L=C igual a _previous_close e V=0
        s.df.loc[_index_new_row] = [_datetime, _O, _H, _L, _C, 0]
        s.df.sort_index(ignore_index=True, inplace=True)
        s.df.reset_index(drop=True)
        s.current_row += 1
        _index_new_row = s.current_row + 0.5
        return _index_new_row

    def is_present_inother_sheets(self, _new_datetime, _nrow_start, _nrow_end):
        s: Sheet
        _datetime_list = []
        for s in self.sheets:
            for i in range(_nrow_start, _nrow_end+1):
                if i > len(s.df) - 1:
                    break
                row = s.df.iloc[i]
                _datetime_str = row['DATETIME']
                _datetime = datetime.fromisoformat(_datetime_str)
                _datetime_list.append(_datetime)

        _ret = _new_datetime in _datetime_list
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
        _datetime_set = set()
        _len_set = set()
        s: Sheet
        for s in self.sheets:
            _datetime_set.add(s.get_datetime_last_row())
            _len_set.add(len(s.df))
        if len(_datetime_set) == 1 and len(_len_set) == 1:
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

    def open_checkpoint(self):
        _filename = 'sync_cp.json'
        if os.path.exists(_filename):
            with open(_filename, 'r') as file:
                self.cp = json.load(file)
            finished = self.cp['finished']
            current_row = self.cp['current_row']
            return True, current_row, finished
        return False, 0, False

    def write_checkpoint(self, finished=False):
        s: Sheet
        _rows_set = set()
        for s in self.sheets:
            _rows_set.add(s.current_row)
        if not self.all_sheets_row_synced():
            print(f'cuidado. write_checkpoint. as planilhas NÃO estão todas sincronizadas ainda. '
                  f'current_rows = {list(_rows_set)}\n')

        _current_row = list(_rows_set)[0]
        self.cp['id'] = ''
        self.cp['finished'] = finished
        self.cp['current_row'] = _current_row
        self.cp['timeframe'] = self.timeframe

        if finished:
            s: Sheet = self.sheets[0]
            row = s.df.iloc[0]
            _datetime_start_str = row['DATETIME']
            self.cp['start'] = _datetime_start_str
            row = s.df.iloc[-1]
            _datetime_end_str = row['DATETIME']
            self.cp['end'] = _datetime_end_str
            _datetime_start_str = _datetime_start_str.replace('-', '').replace(':', '')
            _datetime_end_str = _datetime_end_str.replace('-', '').replace(':', '')
            self.cp['id'] = f'{len(self.symbols)}_{self.timeframe}_{_datetime_start_str}_{_datetime_end_str}'

        self.cp['n_symbols'] = len(self.symbols)
        self.cp['symbols_to_sync'] = self.symbols

        _filename = 'sync_cp.json'
        with open(_filename, 'w') as file:
            json.dump(self.cp, file, indent=4)
        print(f'checkpoint {_filename} gravado. linha atual = {_current_row}\n')

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
            _datetime_str = row['DATETIME']
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
            _datetime_str = row['DATETIME']
            _datetime_previous = datetime.fromisoformat(_datetime_str)
            for i in range(1, len(s.df)):
                row = s.df.iloc[i]
                _datetime_str = row['DATETIME']
                _datetime_current = datetime.fromisoformat(_datetime_str)
                if _datetime_current <= _datetime_previous:
                    print(f'erro em {s.symbol} {_datetime_current}')
                _datetime_previous = _datetime_current

    def append_rows_until(self, s: Sheet, _new_current_row: int):
        _row = s.df.iloc[s.current_row]
        _datetime_str = _row['DATETIME']
        _datetime = datetime.fromisoformat(_datetime_str)

        _previous_close = _row['CLOSE']

        # faz as inserções de novas linhas até _datetime. os datetime's das linhas inseridas
        # começam em _lower_datetime e vão até (mas não incluindo) _datetime.
        _new_date_time = _datetime + s.timedelta
        _index_new_row = s.current_row + 0.5

        while s.current_row < _new_current_row:
            is_present = self.is_present_inother_sheets(_new_date_time,
                                                        s.current_row,
                                                        _new_current_row)
            if not is_present:
                _new_date_time += s.timedelta
                continue

            _index_new_row = self.append_new_row(_index_new_row, _new_date_time, _previous_close, s)
            _new_date_time += s.timedelta


def synchronize():
    settings = read_json('settings.json')
    temp_dir = settings['temp_dir']
    csv_s_dir = settings['csv_s_dir']
    timeframe = settings['timeframe']

    # se o diretório temp não existe, então crie-o.
    if not os.path.exists(temp_dir):
        print('o diretório temp não existe. criando-o.')
        os.mkdir(temp_dir)
        _filename = f'{temp_dir}/.directory'
        _f = open(_filename, 'x')  # para manter o diretório no git
        _f.close()

    symbols = search_symbols_in_directory(temp_dir, timeframe)
    _len_symbols = len(symbols)
    if _len_symbols == 0:
        print('Não há arquivos CSVs para serem sincronizados.')
        exit(-1)
    elif _len_symbols == 1:
        print('Há apenas 1 arquivo CSV. Portanto, o arquivo será considerado já sincronizado.')
        make_backup(temp_dir, csv_s_dir)
        exit(0)

    dir_sync = DirectorySynchronization(temp_dir)
    dir_sync.synchronize_directory()


if __name__ == '__main__':
    synchronize()
