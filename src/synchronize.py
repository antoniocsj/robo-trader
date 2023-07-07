import json
from numpy import ndarray
from datetime import datetime
from Sheet import Sheet
from HistMulti import HistMulti
from utils_filesystem import read_json
from utils_symbols import search_symbols_in_dict
from utils_nn import prepare_data_for_prediction


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
        self.sheets = {}
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

    # def load_sheets_(self):
    #     for _symbol in self.symbols:
    #         d = self.symbols_rates[f'{_symbol}_{self.timeframe}']
    #         self.sheets.append(Sheet(d, _symbol, self.timeframe))

    def load_sheets(self):
        for _symbol in self.symbols:
            key = f'{_symbol}_{self.timeframe}'
            d = self.symbols_rates[key]
            self.sheets[key] = Sheet(d, _symbol, self.timeframe)

    def check_is_trading(self, s: Sheet):
        row = s.df.iloc[-1]
        _datetime_str = row['DATETIME']
        _datetime = datetime.fromisoformat(_datetime_str)
        if _datetime + s.timedelta > self.server_datetime:
            s.is_trading = True
        else:
            s.is_trading = False

    def synchronize_symbols(self):
        s: Sheet
        print(f'server_datetime = {self.server_datetime}')

        # verificar quais símbolos estão (ou não) operando
        for k in self.sheets:
            s = self.sheets[k]
            self.check_is_trading(s)
            print(f'{s.symbol} is trading: {s.is_trading}')

        for k in self.sheets:
            s = self.sheets[k]
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


def synchronize(data: dict) -> ndarray:
    """
    Prepara os dados históricos para seu uso no modelo (rede neural). Faz todos os ajustes necessários para retornar
    um array pronto para ser apresentado ao modelo para obter uma previsão.
    Entre os ajustes estão a remoção de todos os símbolos desnessários, pois a requisição pode possuir um conjunto de
    símbolos maior do que aquele que foi usado no treinamento da rede neural.
    Outro ajuste importante é a remoção da última vela que pode estar em formação, no caso dos símbolos que estão
    operando.
    Outro ajuste é feito nas velas dos símbolos que não estão operando. Nessas velas é feito O=H=L=C=C' e V=0.
    Também deverá ser implementada a sincronização de todos os símbolos.
    :param data: dados históricos provenientes de uma requisição feita pelo MT5.
    :return: array pronto para ser aplicado no modelo
    """
    print('synchronize()')

    setup = read_json('setup.json')
    csv_dir = setup['csv_dir']
    symbol_out = setup['symbol_out']
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
    symbols_present_in_the_request = set(symbols)

    with open("train_configs.json", "r") as file:
        train_configs = json.load(file)

    print(f'train_configs:')
    print(f'{train_configs}')

    n_steps = train_configs['n_steps']
    n_features = train_configs['n_features']
    symbol_out = train_configs['symbol_out']
    symbols_used_in_training = set(train_configs['symbols'])
    n_samples_train = train_configs['n_samples_train']
    tipo_vela = train_configs['tipo_vela']

    # verifique se os símbolos usados no treinamento da rede neural estão presentes na requisição
    if symbols_used_in_training.issubset(symbols_present_in_the_request):
        # faça um novo symbols_rates contendo apenas os símbolos presentes no treinamento
        _new_symbol_rates = {}
        symbols_used_in_training = train_configs['symbols']  # use a lista, pois é garantido que está ordenada.
        for _symbol in symbols_used_in_training:
            key = f'{_symbol}_{timeframe}'
            _new_symbol_rates[key] = symbols_rates[key]
        symbols_rates = _new_symbol_rates
    else:
        print(f'ERRO. Nem todos os símbolos usados no treinamento da rede neural estão presentes na requisição.')
        exit(-1)

    symb_sync = SymbolsSynchronization(symbols_rates, timeframe, trade_server_datetime, n_steps)
    symb_sync.synchronize_symbols()
    hist = HistMulti(symb_sync.sheets, timeframe)

    # ainda pode faltar a normalização ou outras operações antes de usar os dados históricos no modelo para previsão.
    # depende do setup usado.

    x_input = prepare_data_for_prediction(hist, n_steps, tipo_vela)
    x_input = x_input.reshape((1, n_steps, n_features))

    return x_input


def test_01():
    from keras.models import load_model

    with open("train_configs.json", "r") as file:
        train_configs = json.load(file)

    data = read_json('request_3.json')
    x_input = synchronize(data)

    model = load_model('model.h5')

    close_pred_norm = model.predict(x_input)

    bias = train_configs['bias']
    # close_pred_denorm = denorm_close_price(close_pred_norm[0][0] + bias, trans)


if __name__ == '__main__':
    test_01()
