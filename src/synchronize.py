from datetime import datetime
from utils import read_json
from Sheet import SheetRates


def search_symbols_in_dict(_dict: dict, timeframe: str) -> list[str]:
    """
    Procurando pelos símbolos presentes num dicionário contendo velas de vários ativos.
    Todos os arquivos devem ser do mesmo timeframe.
    :return: lista dos símbolos
    """
    # passe por todos as chaves do dicionário e descubra o symbol e timeframe
    symbols = []

    for symbol_tf in _dict:
        _symbol = symbol_tf.split('_')[0]
        _timeframe = symbol_tf.split('_')[1]

        if _symbol not in symbols:
            symbols.append(_symbol)
        else:
            print(f'erro. o símbolo {_symbol} aparece repetido.')
            exit(-1)

        if _timeframe != timeframe:
            print(f'ERRO. o timeframe {_timeframe} é diferente do especificado {timeframe}.')
            exit(-1)

    symbols = sorted(symbols)
    return symbols


class SymbolsSynchronization:
    """
    Realiza a correção/sincronização de todos os símbolos contidos no dicionário.
    Quando realiza inserções de linhas, faz do seguinte modo:
    - Obtém o valor de fechamento da última vela conhecida (C');
    - as velas inseridas terão O=H=L=C=C' e V=0;
    Todas as planilhas são sincronizadas simultaneamente em apenas 1 processo.
    """

    def __init__(self, symbols_rates: dict, timeframe: str, server_datetime: datetime, n_steps: int):
        self.symbols_rates = symbols_rates
        self.symbols = []
        self.timeframe = timeframe
        self.server_datetime = server_datetime
        self.sheets = []
        self.num_insertions_done = 0
        self.exclude_first_rows = False
        self.new_start_row_datetime: datetime = None
        self.n_steps = n_steps

        self.search_symbols()
        self.load_sheets()

    def search_symbols(self):
        """
        Procurando pelos arquivos csv correspondentes ao 'simbolo' e ao 'timeframe'
        :return:
        """
        # passe por todos as chaves do dicionário e descubra o symbol e timeframe
        symbols = []

        for symbol_tf in self.symbols_rates:
            _symbol = symbol_tf.split('_')[0]
            _timeframe = symbol_tf.split('_')[1]

            if _symbol not in symbols:
                symbols.append(_symbol)
            else:
                print(f'erro. o símbolo {_symbol} aparece repetido.')
                exit(-1)

            if _timeframe != self.timeframe:
                print(f'ERRO. o timeframe {_timeframe} é diferente do especificado {self.timeframe}.')
                exit(-1)

        self.symbols = sorted(symbols)

    def load_sheets(self):
        for _symbol in self.symbols:
            d = self.symbols_rates[f'{_symbol}_{self.timeframe}']
            self.sheets.append(SheetRates(d, _symbol, self.timeframe))

    def check_is_trading(self, s: SheetRates):
        row = s.df.iloc[-1]
        _datetime_str = row['DATETIME']
        _datetime = datetime.fromisoformat(_datetime_str)
        if _datetime + s.timedelta > self.server_datetime:
            s.is_trading = True
        else:
            s.is_trading = False

    def synchronize_symbols(self):
        s: SheetRates
        print(f'server_datetime = {self.server_datetime}')

        # verificar quais símbolos estão (ou não) operando
        for s in self.sheets:
            self.check_is_trading(s)
            print(f'{s.symbol} is trading: {s.is_trading}')

        for s in self.sheets:
            if s.is_trading:
                # se está operando, então descarte a vela em formação (última)
                # e mantenha apenas as N (n_steps) últimas velas.
                if len(s.df) < self.n_steps + 1:
                    print(f'o tamanho da planilha {s.symbol} ({len(s.df)}) é menor do que n_steps + 1 '
                          f'({self.n_steps + 1})')
                    exit(-1)
                s.df.drop(s.df.index[-1], inplace=True)
                s.df.sort_index(ignore_index=True, inplace=True)
                s.df.reset_index(drop=True)
                s.df.drop(s.df.index[:-self.n_steps], inplace=True)
                s.df.sort_index(ignore_index=True, inplace=True)
                s.df.reset_index(drop=True)
            else:
                # se não está operando, então todas as vela são velas concluídas e antigas;
                # mantenha apenas as (n_steps) últimas velas;
                # obtenha o preço de fechamento da última vela (C') e aplique nos OHLCs da (n_steps) últimas velas;
                if len(s.df) < self.n_steps:
                    print(f'o tamanho da planilha {s.symbol} ({len(s.df)}) é menor do que n_steps '
                          f'({self.n_steps})')
                    exit(-1)
                s.df.drop(s.df.index[:-self.n_steps], inplace=True)
                s.df.sort_index(ignore_index=True, inplace=True)
                s.df.reset_index(drop=True)

                _close = s.df.iloc[-1]['CLOSE']
                for i in range(self.n_steps - 1, -1, -1):
                    s.df.loc[i, 'OPEN'] = _close
                    s.df.loc[i, 'HIGH'] = _close
                    s.df.loc[i, 'LOW'] = _close
                    s.df.loc[i, 'CLOSE'] = _close
                    s.df.loc[i, 'TICKVOL'] = 0


def synchronize(data: dict):
    """
    Sincroniza as velas.
    :param data:
    :return:
    """
    print('synchronize()')

    setup = read_json('setup.json')
    csv_dir = setup['csv_dir']
    setup_timeframe = setup['timeframe']

    last_datetime = datetime.fromisoformat(data['last_datetime'])
    trade_server_datetime = datetime.fromisoformat(data['trade_server_datetime'])
    print(f'last_datetime = {last_datetime}, trade_server_datetime = {trade_server_datetime}')

    timeframe = data['timeframe']
    if timeframe != setup_timeframe:
        print(f'o timeframe da requisição ({timeframe}) é diferente do timeframe definido no arquivo '
              f'setup.json ({setup_timeframe})')
        exit(-1)

    n_symbols = data['n_symbols']
    rates_count = data['rates_count']
    start_pos = data['start_pos']
    print(f'timeframe = {timeframe}, n_symbols = {n_symbols}, '
          f'rates_count = {rates_count}, start_pos = {start_pos} ')

    symbols_rates = data['symbols_rates']
    symbols = search_symbols_in_dict(symbols_rates, timeframe)
    _len_symbols = len(symbols)
    if _len_symbols == 0:
        print('ERRO. Não há símbolos a serem sincronizados.')
        exit(-1)
    elif _len_symbols == 1:
        print('Há apenas 1 símbolo. Portanto, ele será considerado sincronizado.')
        return

    n_steps = 2
    symb_sync = SymbolsSynchronization(symbols_rates, timeframe, trade_server_datetime, n_steps)
    symb_sync.synchronize_symbols()
    pass


def test_01():
    data = read_json('request_3.json')
    synchronize(data)


if __name__ == '__main__':
    test_01()
