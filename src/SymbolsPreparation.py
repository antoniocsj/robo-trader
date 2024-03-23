from datetime import datetime
from Sheet import Sheet


class SymbolsPreparation:
    """
    Realiza a preparação/correção/sincronização de todos os símbolos contidos no dicionário.
    Quando realiza inserções de linhas, faz do seguinte modo:
    - Obtém o valor de fechamento da última vela conhecida (C');
    - as velas inseridas terão O=H=L=C=C' e V=0;
    Todas as planilhas são preparadas/sincronizadas simultaneamente em apenas 1 processo.
    """

    def __init__(self, symbols_rates: dict, timeframe: str, server_datetime: datetime, n_steps: int):
        self.symbols_rates = symbols_rates
        self.symbols = []
        self.timeframe = timeframe
        self.server_datetime = server_datetime
        self.sheets = {}
        self.num_insertions_done = 0
        self.exclude_first_rows = False
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

        for symbol in self.symbols_rates:
            # _symbol = symbol_tf.split('_')[0]
            # _timeframe = symbol_tf.split('_')[1]

            if symbol not in symbols:
                symbols.append(symbol)
            else:
                print(f'erro. o símbolo {symbol} aparece repetido.')
                exit(-1)

            # if _timeframe != self.timeframe:
            #     print(f'ERRO. o timeframe {_timeframe} é diferente do especificado {self.timeframe}.')
            #     exit(-1)

        self.symbols = sorted(symbols)

    # def load_sheets_(self):
    #     for _symbol in self.symbols:
    #         d = self.symbols_rates[f'{_symbol}_{self.timeframe}']
    #         self.sheets.append(Sheet(d, _symbol, self.timeframe))

    def load_sheets(self):
        for symbol in self.symbols:
            d = self.symbols_rates[symbol][self.timeframe]
            self.sheets[symbol] = {}
            self.sheets[symbol][self.timeframe] = Sheet(d, symbol, self.timeframe)

    def check_is_trading(self, s: Sheet):
        row = s.df.iloc[-1]
        _datetime_str = row['DATETIME']
        _datetime = datetime.fromisoformat(_datetime_str)
        if _datetime + s.timedelta > self.server_datetime:
            s.is_trading = True
        else:
            s.is_trading = False

    def prepare_symbols(self, all_symbols_trading=False):
        # verificar quais símbolos estão (ou não) operando
        who_is_trading = []
        for symbol_name in self.sheets:
            sheet: Sheet = self.sheets[symbol_name][self.timeframe]

            if all_symbols_trading:
                sheet.is_trading = True
            else:
                self.check_is_trading(sheet)

            if sheet.is_trading:
                who_is_trading.append(sheet.symbol)

        if len(who_is_trading) == 0:
            print('nenhum símbolo está operando agora.')
            # exit(-1)

        if len(who_is_trading) == len(self.symbols):
            print('TODOS os símbolos estão operando.')
        else:
            print('NEM todos os símbolos estão operando.')

        for symbol_name in self.sheets:
            sheet: Sheet = self.sheets[symbol_name][self.timeframe]
            if sheet.is_trading:
                # se está operando, então descarte a vela em formação (última)
                # e mantenha apenas as N (n_steps) últimas velas.
                if len(sheet.df) < self.n_steps + 1:
                    print(f'o tamanho da planilha {sheet.symbol} ({len(sheet.df)}) é menor do que n_steps + 1 '
                          f'({self.n_steps + 1})')
                    exit(-1)
                sheet.df.drop(sheet.df.index[-1], inplace=True)
                sheet.df.sort_index(ignore_index=True, inplace=True)
                sheet.df.reset_index(drop=True)
                sheet.df.drop(sheet.df.index[:-self.n_steps], inplace=True)
                sheet.df.sort_index(ignore_index=True, inplace=True)
                sheet.df.reset_index(drop=True)
            else:
                # se não está operando, então todas as vela são velas concluídas e antigas;
                # mantenha apenas as (n_steps) últimas velas;
                # obtenha o preço de fechamento da última vela (C') e aplique nos OHLCs da (n_steps) últimas velas;
                if len(sheet.df) < self.n_steps:
                    print(f'o tamanho da planilha {sheet.symbol} ({len(sheet.df)}) é menor do que n_steps '
                          f'({self.n_steps})')
                    exit(-1)
                sheet.df.drop(sheet.df.index[:-self.n_steps], inplace=True)
                sheet.df.sort_index(ignore_index=True, inplace=True)
                sheet.df.reset_index(drop=True)

                _close = sheet.df.iloc[-1]['CLOSE']
                for i in range(self.n_steps - 1, -1, -1):
                    sheet.df.loc[i, 'OPEN'] = _close
                    sheet.df.loc[i, 'HIGH'] = _close
                    sheet.df.loc[i, 'LOW'] = _close
                    sheet.df.loc[i, 'CLOSE'] = _close
                    sheet.df.loc[i, 'TICKVOL'] = 0
