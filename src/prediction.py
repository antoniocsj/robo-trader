import json
import pickle
import numpy as np
from numpy import ndarray
from datetime import datetime
from Sheet import Sheet
from HistMulti import HistMulti
from utils_filesystem import read_json
from utils_symbols import search_symbols_in_dict
from utils_nn import prepare_data_for_prediction
from utils_ops import denorm_output
from setups import apply_setup_symbols


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

    def prepare_symbols(self):
        s: Sheet
        print(f'server_datetime = {self.server_datetime}')

        # verificar quais símbolos estão (ou não) operando
        _who_is_trading = []
        for s in list(self.sheets.values()):
            self.check_is_trading(s)
            print(f'{s.symbol} is trading: {s.is_trading}')
            if s.is_trading:
                _who_is_trading.append(s.symbol)

        if len(_who_is_trading) == 0:
            print('nenhum símbolo está operando agora.')
            exit(-1)

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


def prepare_data_for_model(data: dict) -> ndarray:
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
    print('prepare_data_for_model()')

    settings = read_json('settings.json')
    print('settings:')
    print(f'{settings}')

    temp_dir = settings['temp_dir']
    symbol_out = settings['symbol_out']
    settings_timeframe = settings['timeframe']
    setup_code = settings['setup_code']
    setup_uses_differentiation = settings['setup_uses_differentiation']

    train_configs = read_json('train_config.json')
    print('train_configs:')
    print(f'{train_configs}')

    n_steps = train_configs['n_steps']
    n_features = train_configs['n_features']
    symbol_out = train_configs['symbol_out']
    symbols_used_in_training: list[str] = train_configs['symbols']
    n_samples_train = train_configs['n_samples_train']
    candle_input_type = settings['candle_input_type']
    candle_output_type = settings['candle_output_type']

    if setup_code < 1:
        print(f'ERRO. setup_code = {setup_code} indica que não foi feito nenhum setup.')
        exit(-1)

    if setup_uses_differentiation:
        num_candles = n_steps + 1
    else:
        num_candles = n_steps

    last_datetime = datetime.fromisoformat(data['last_datetime'])
    trade_server_datetime = datetime.fromisoformat(data['trade_server_datetime'])
    print(f'last_datetime = {last_datetime}, trade_server_datetime = {trade_server_datetime}')

    timeframe = data['timeframe']
    if timeframe != settings_timeframe:
        print(f'o timeframe da requisição ({timeframe}) é diferente do timeframe definido no arquivo '
              f'settings.json ({settings_timeframe})')
        exit(-1)

    n_symbols = data['n_symbols']
    rates_count = data['rates_count']
    start_pos = data['start_pos']
    print(f'timeframe = {timeframe}, n_symbols = {n_symbols}, '
          f'rates_count = {rates_count}, start_pos = {start_pos} ')

    if num_candles > rates_count - 1:
        print(f'ERRO. o número de velas presentes na requisição não é suficiente para o modelo.')
        exit(-1)

    symbols_rates = data['symbols_rates']
    symbols = search_symbols_in_dict(symbols_rates, timeframe)
    symbols_present_in_the_request_set = set(symbols)

    # deixe na lista pure_symbols_used_in_training_set apenas os símbolos puros, ou seja, não modificados
    # (sem '@' no nome)
    pure_symbols_used_in_training_set = set()
    name: str
    for name in symbols_used_in_training:
        index = name.find('@')
        if index == -1:
            pure_name = name
        else:
            pure_name = name[0:index]
        pure_symbols_used_in_training_set.add(pure_name)

    # verifique se os símbolos usados no treinamento da rede neural estão presentes na requisição
    if pure_symbols_used_in_training_set.issubset(symbols_present_in_the_request_set):
        # faça um novo symbols_rates contendo apenas os símbolos presentes no treinamento
        _new_symbol_rates = {}
        pure_symbols_used_in_training = sorted(list(pure_symbols_used_in_training_set))
        for _symbol in pure_symbols_used_in_training:
            key = f'{_symbol}_{timeframe}'
            _new_symbol_rates[key] = symbols_rates[key]
        symbols_rates = _new_symbol_rates
    else:
        print(f'ERRO. Nem todos os símbolos usados no treinamento da rede neural estão presentes na requisição.')
        exit(-1)
    
    symb_sync = SymbolsPreparation(symbols_rates, timeframe, trade_server_datetime, num_candles)
    symb_sync.prepare_symbols()
    hist = HistMulti(symb_sync.sheets, timeframe)
    hist2 = apply_setup_symbols(hist, setup_code)

    if hist2.symbols != symbols_used_in_training:
        print(f'ERRO. hist2.symbols != symbols_used_in_training.')
        exit(-1)

    X = prepare_data_for_prediction(hist2, n_steps, candle_input_type)
    X = np.asarray(X).astype(np.float32)
    X = X.reshape((1, n_steps, n_features))

    # for MLP model only
    # n_input = X.shape[1] * X.shape[2]
    # X = X.reshape((X.shape[0], n_input))

    return X


def predict_next_candle(data: dict):
    from keras.models import load_model

    settings = read_json('settings.json')
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    _symbol_tf = f'{symbol_out}_{timeframe}'

    train_config = read_json('train_config.json')

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    x_input = prepare_data_for_model(data)

    model = load_model('model.h5')
    output_norm = model.predict(x_input)

    bias = train_config['bias']
    candle_output_type = train_config['candle_output_type']
    scaler = scalers[_symbol_tf]

    print('considerando o bias(+):')
    output_denorm = denorm_output(output_norm, bias, candle_output_type, scaler)
    print(f'previsão para a próxima vela: {candle_output_type} = {output_denorm}')

    if candle_output_type == 'OHLC' or candle_output_type == 'OHLCV':
        dCO = output_denorm[3] - output_denorm[0]
        print(f'C - O = {dCO}')

    print('considerando o bias=0:')
    output_denorm = denorm_output(output_norm, 0.0, candle_output_type, scaler)
    print(f'previsão para a próxima vela: {candle_output_type} = {output_denorm}')

    if candle_output_type == 'OHLC' or candle_output_type == 'OHLCV':
        dCO = output_denorm[3] - output_denorm[0]
        print(f'C - O = {dCO}')

    print('considerando o bias(-):')
    bias = (-np.array(bias)).tolist()
    output_denorm = denorm_output(output_norm, bias, candle_output_type, scaler)
    print(f'previsão para a próxima vela: {candle_output_type} = {output_denorm}')

    if candle_output_type == 'OHLC' or candle_output_type == 'OHLCV':
        dCO = output_denorm[3] - output_denorm[0]
        print(f'C - O = {dCO}')


def test_01():
    data = read_json('request_1.json')
    predict_next_candle(data)


if __name__ == '__main__':
    test_01()
