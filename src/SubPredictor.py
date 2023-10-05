import os
import pickle
import numpy as np
from numpy import ndarray
from datetime import datetime
from HistMulti import HistMulti
from utils_filesystem import read_json
from utils_symbols import search_symbols_in_dict
from utils_nn import prepare_data_for_prediction
from utils_ops import denorm_output
from setups import apply_setup_symbols
from keras.models import load_model
from prediction import SymbolsPreparation


def get_timeframe_in_minutes(tf: str) -> int:
    if tf == 'M1':
        ret = 1
    elif tf == 'M2':
        ret = 2
    elif tf == 'M3':
        ret = 3
    elif tf == 'M4':
        ret = 4
    elif tf == 'M5':
        ret = 5
    elif tf == 'M6':
        ret = 6
    elif tf == 'M10':
        ret = 10
    elif tf == 'M12':
        ret = 12
    elif tf == 'M15':
        ret = 15
    elif tf == 'M20':
        ret = 20
    elif tf == 'M30':
        ret = 30
    elif tf == 'H1':
        ret = 1 * 60
    elif tf == 'H2':
        ret = 2 * 60
    elif tf == 'H3':
        ret = 3 * 60
    elif tf == 'H4':
        ret = 4 * 60
    elif tf == 'H6':
        ret = 6 * 60
    elif tf == 'H8':
        ret = 8 * 60
    elif tf == 'H12':
        ret = 12 * 60
    elif tf == 'D1':
        ret = 24 * 60
    elif tf == 'W1':
        ret = 7 * 24 * 60
    else:
        print(f'ERRO. timeframe inválido. ({tf})')
        exit(-1)

    return ret


class SubPredictor:
    def __init__(self, _id: str, directory: str):
        self.id = _id
        self.directory = directory
        self.settings = None
        self.train_config = None
        self.scalers = None
        self.model = None
        self.output = None
        self.symbol_out = ''
        self.timeframe = ''
        self.timeframe_in_minutes = 0
        self.n_steps = 0
        self.candle_input_type = ''
        self.candle_output_type = ''
        self.n_hidden_layers = 0
        self.n_samples_train = 0
        self.whole_set_train_loss_eval = 0.0
        self.n_samples_test = 0
        self.test_loss_eval = 0.0
        self.loss = 0.0

        self.load()

    def load(self):
        # directory = f'../predictors/{self.id:02d}'
        directory = f'{self.directory}/{self.id}'
        if not os.path.exists(directory):
            print(f'ERRO. o diretório {directory} não existe.')
            exit(-1)

        settings_filepath = f'{directory}/settings.json'
        if not os.path.exists(settings_filepath):
            print(f'ERRO. o arquivo {settings_filepath} não existe.')
            exit(-1)
        self.settings = read_json(settings_filepath)

        train_config_filepath = f'{directory}/train_config.json'
        if not os.path.exists(train_config_filepath):
            print(f'ERRO. o arquivo {train_config_filepath} não existe.')
            exit(-1)
        self.train_config = read_json(train_config_filepath)

        scalers_filepath = f'{directory}/scalers.pkl'
        if not os.path.exists(scalers_filepath):
            print(f'ERRO. o arquivo {scalers_filepath} não existe.')
            exit(-1)
        with open(scalers_filepath, 'rb') as file:
            self.scalers = pickle.load(file)

        model_filepath = f'{directory}/model.h5'
        if not os.path.exists(model_filepath):
            print(f'ERRO. o arquivo {model_filepath} não existe.')
            exit(-1)
        self.model = load_model(model_filepath)

        self.symbol_out: str = self.train_config['symbol_out']
        self.timeframe: str = self.train_config['timeframe']
        self.timeframe_in_minutes: int = get_timeframe_in_minutes(self.timeframe)
        self.n_steps: int = self.train_config['n_steps']
        self.candle_input_type: str = self.train_config['candle_input_type']
        self.candle_output_type: str = self.train_config['candle_output_type']
        self.n_hidden_layers: int = self.train_config['n_hidden_layers']
        self.n_samples_train: int = self.train_config['n_samples_train']
        self.n_samples_test: int = self.train_config['n_samples_test']
        self.whole_set_train_loss_eval = self.train_config['whole_set_train_loss_eval']
        self.test_loss_eval: float = self.train_config['test_loss_eval']
        self.loss: float = self.whole_set_train_loss_eval * self.test_loss_eval

    def prepare_data_for_model(self, data: dict, force_all_symbols_trading=False) -> ndarray:
        """
        Prepara os dados históricos para seu uso no modelo (rede neural). Faz todos os ajustes necessários para retornar
        um array pronto para ser apresentado ao modelo para obter uma previsão.
        Entre os ajustes estão a remoção de todos os símbolos desnessários, pois a requisição pode possuir um conjunto
        de símbolos maior do que aquele que foi usado no treinamento da rede neural.
        Outro ajuste importante é a remoção da última vela que pode estar em formação, no caso dos símbolos que estão
        operando.
        Outro ajuste é feito nas velas dos símbolos que não estão operando. Nessas velas é feito O=H=L=C=C' e V=0.
        Também deverá ser implementada a sincronização de todos os símbolos.
        :param force_all_symbols_trading: supõe que todos os símbolos estão operando no instante da previsão.
        :param data: dados históricos provenientes de uma requisição feita pelo MT5.
        :return: array pronto para ser aplicado no modelo
        """
        settings = self.settings
        scalers = self.scalers

        # print('settings:')
        # print(f'{settings}')

        timeframe = settings['timeframe']
        setup_code = settings['setup_code']
        setup_uses_differentiation = settings['setup_uses_differentiation']

        train_configs = self.train_config
        # print('train_configs:')
        # print(f'{train_configs}')

        n_steps = train_configs['n_steps']
        n_features = train_configs['n_features']
        symbols_used_in_training: list[str] = train_configs['symbols']
        candle_input_type = settings['candle_input_type']

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

        timeframes = data['timeframes']
        if timeframe not in timeframes:
            print(f'ERRO! O timeframe do predictor ({timeframe})  não está presente na requisição.')
            exit(-1)

        rates_count = data['rates_count']

        if num_candles > rates_count - 1:
            print(f'ERRO. o número de velas presentes na requisição não é suficiente para o modelo.')
            exit(-1)

        symbols_rates = data['symbols_rates']
        symbols = search_symbols_in_dict(symbols_rates)
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
            for symbol in pure_symbols_used_in_training:
                # key = f'{symbol}_{timeframe}'
                # _new_symbol_rates[key] = symbols_rates[key]
                _new_symbol_rates[symbol] = {}
                _new_symbol_rates[symbol][timeframe] = symbols_rates[symbol][timeframe]
            symbols_rates = _new_symbol_rates
        else:
            print(f'ERRO. Nem todos os símbolos usados no treinamento da rede neural estão presentes na requisição.')
            exit(-1)

        symb_sync = SymbolsPreparation(symbols_rates, timeframe, trade_server_datetime, num_candles)

        if force_all_symbols_trading:
            symb_sync.prepare_symbols(all_symbols_trading=True)
        else:
            symb_sync.prepare_symbols()

        hist = HistMulti(symb_sync.sheets, timeframe)
        hist2 = apply_setup_symbols(hist, setup_code, settings, scalers)

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

    def calc_output(self, input_data: dict, force_all_symbols_trading=False):
        symbol_out = self.settings['symbol_out']
        timeframe = self.settings['timeframe']
        _symbol_tf = f'{symbol_out}_{timeframe}'

        x_input = self.prepare_data_for_model(input_data, force_all_symbols_trading)
        output_norm = self.model.predict(x_input)

        bias = self.train_config['bias']
        candle_output_type = self.train_config['candle_output_type']
        scaler = self.scalers[_symbol_tf]

        output_denorm = denorm_output(output_norm, bias, candle_output_type, scaler)
        self.output = output_denorm

    def show_output(self):
        candle_output_type = self.train_config['candle_output_type']
        # directory = self.directory.split('/')[1]

        print(f'sub_predictor ({self.id}) {self.candle_input_type} S={self.n_steps} '
              f'HL={self.n_hidden_layers} L={self.loss:.3e}: ', end='')
        if len(candle_output_type) == 1:
            if self.train_config['symbol_out'] == 'XAUUSD':
                print(f'{candle_output_type} = {self.output:.2f}')
            else:
                print(f'{candle_output_type} = {self.output:.5f}')
        else:
            print(f'{candle_output_type} = {self.output}')


def teste_01():
    data = read_json('request.json')
    suppose_all_symbols_trading = True

    directory = '../predictors/M10A'
    predictor_1 = SubPredictor('M10_OHLC_S2_HL1', directory)
    predictor_1.calc_output(data, suppose_all_symbols_trading)
    predictor_1.show_output()


if __name__ == '__main__':
    teste_01()
