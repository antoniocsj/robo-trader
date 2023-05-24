import os
from datetime import datetime, timedelta
import pandas as pd
from pandas import DataFrame


class Sheet:
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

    def print_current_row(self):
        row = self.df.iloc[self.current_row]
        _date, _time = row['<DATE>'], row['<TIME>']
        _O, _H, _L, _C, _V = row['<OPEN>'], row['<HIGH>'], row['<LOW>'], row['<CLOSE>'], row['<TICKVOL>']
        print(f'{self.symbol} {_date} {_time} OHLCV = {_O} {_H} {_L} {_C} {_V}')


class DirectoryCorrection:
    def __init__(self, directory: str):
        self.directory = directory
        self.all_files = []
        self.csv_files = {}
        self.symbols = []
        self.timeframe = ''
        self.sheets = []
        self.first_insertion_done = False
        self.new_start_row_datetime: datetime = None

        self.search_symbols()
        self.load_sheets()

    def search_symbols(self):
        """
        Procurando pelos arquivos csv correspondentes ao 'simbolo' e ao 'timeframe'
        :return:
        """
        # antes de tudo, remova todos os arquivos corrigidos, pois uma nova correção será feita.
        self.all_files = os.listdir(self.directory)
        for filename in self.all_files:
            if filename.endswith('_C.csv'):
                _filepath = self.directory + '/' + filename
                os.remove(_filepath)

        # passe por todos os arquivos csv e descubra o symbol e timeframe
        self.all_files = os.listdir(self.directory)
        for filename in self.all_files:
            if filename.endswith('.csv'):
                _symbol = filename.split('_')[0]
                _timeframe = filename.split('_')[1]

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

    def load_sheets(self):
        for i, k in enumerate(self.csv_files):
            _symbol, _timeframe = k.split('_')
            _filepath = self.get_csv_filepath(k)
            self.sheets.append(Sheet(_filepath, _symbol, _timeframe))

    def find_first_row(self):
        print('find_first_row')

        # assegure que a primeira linha de todas as planilhas contém a mesma data.
        s: Sheet
        _dates_set = set()
        for s in self.sheets:
            df: DataFrame = s.df
            _date = df.iloc[0]['<DATE>']
            _dates_set.add(_date)

        print(f'datas encontradas na primeira linha de todas as planilhas: {_dates_set}')
        if len(_dates_set) > 1:
            print('erro. nem todas as planilhas começam na mesma data.')
            exit(-1)

        # analise os horários presentes na primeira linha de todas as planilhas.
        _times_set = set()
        for s in self.sheets:
            df = s.df
            _time = df.iloc[0]['<TIME>']
            _times_set.add(_time)

        print(f'horários encontrados na primeira linha de todas as planilhas: {_times_set}')
        if len(_times_set) > 1:
            print('nem todas as planilhas começam no mesmo horário.')
        else:
            # se todas as planilhas começam na mesma data e horário, então apenas retorne.
            # pois todas as planilhas já tem o current_row ajustado para 0 inicialmente.
            print('todas as planilhas começam no mesmo horário.')
            return

        # buscando uma nova linha que será o novo começo.
        # avance para a 1a vela do próximo dia útil válido.
        # para que uma data ou horário sejam válidos, tem que estar presente em todas as planilhas.
        for s in self.sheets:
            s.go_to_next_day()

        for s in self.sheets:
            s.print_current_row()

    def correct_directory(self):
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
            _results = set()

            self.insert_rows()
            
            for s in self.sheets:
                _r = s.go_to_next_row()
                _results.add(_r)

            if _counter % 10000 == 0:
                print(f'{100 * _counter / _max_len: .2f} %')

            if len(_results) == 1 and list(_results)[0] is False:
                break

            _counter += 1

        for s in self.sheets:
            s.print_current_row()

    def insert_rows(self):
        """
        Se for necessário, faz inserções de linhas nas planilhas.
        :return:
        """
        # verifique se a linha atual de cada planilha contém a mesma data e horário.
        s: Sheet
        _date_time_list = []
        for s in self.sheets:
            row = s.df.iloc[s.current_row]
            _date = row['<DATE>'].replace('.', '-')
            _time = row['<TIME>']
            _date_time_str = f"{_date}T{_time}"
            _date_time = datetime.fromisoformat(_date_time_str)
            _date_time_list.append(_date_time)

        _date_time_set = set(_date_time_list)
        if len(_date_time_set) == 1:
            # todas as planilhas estão sincronizadas nesta linha.
            return

        # nem todas as planilhas estão sincronizadas nesta linha.
        # descubra qual planilha possui o menor 'datetime'.
        _lower_datetime = min(_date_time_list)

        # o menor datetime será a referência. se alguma planilha tiver um datetime maior,
        # então essa planilha sofrerá inserções de novas linhas até que esteja sincronizada.
        for s in self.sheets:
            _row = s.df.iloc[s.current_row]
            _date = _row['<DATE>'].replace('.', '-')
            _time = _row['<TIME>']
            _date_time_str = f"{_date}T{_time}"
            _date_time = datetime.fromisoformat(_date_time_str)

            if _date_time > _lower_datetime:
                print(f'{s.symbol} {_date_time} > {_lower_datetime}')
                _previous_row = s.df.iloc[s.current_row - 1]
                _previous_close = _previous_row['<CLOSE>']

                # faz as inserções de novas linhas até _date_time. os datetime's das linhas inseridas
                # começam em _lower_datetime e vão até (mas não incluindo) _date_time.
                _new_date_time = _lower_datetime
                self.new_start_row_datetime = _lower_datetime
                _index_new_row = s.current_row - 0.5

                while _new_date_time < _date_time:
                    print(f'inserindo nova linha {_new_date_time}')

                    _date = _new_date_time.strftime('%Y.%m.%d')
                    _time = _new_date_time.strftime('%H:%M:%S')
                    _O = _H = _L = _C = _previous_close

                    # insere a nova linha. que será uma vela com O=H=L=C igual a _previous_close e V=0
                    s.df.loc[_index_new_row] = [_date, _time, _O, _H, _L, _C, 0, 0, 0]
                    s.df.sort_index(ignore_index=True, inplace=True)
                    s.df.reset_index(drop=True)
                    s.current_row += 1
                    _index_new_row = s.current_row - 0.5

                    self.first_insertion_done = True
                    _new_date_time += s.timedelta

        # se estas foram as primeiras inserções sofridas pelas planilhas, então as planilhas estão
        # sincronizadas pelo menos nas linhas que vão de new_start_row_datetime até _date_time.
        # portanto pode excluir todas as linhas anteriores a new_start_row_datetime.
        if self.first_insertion_done:
            print('excluindo todas as linhas iniciais desnecessárias.')


def main():
    dir_cor = DirectoryCorrection('./csv')
    dir_cor.correct_directory()


if __name__ == '__main__':
    main()
