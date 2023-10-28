import pickle
from utils_filesystem import read_json
from HistMulti import HistMulti


def initial_compliance_checks():
    """
    Realiza as verificações iniciais de conformidade:
    -> o conteúdo de temp, scalers.pkl deve estar em conformidade com params_nn.json e settings.json
    :return:
    """
    params_nn = read_json('params_nn.json')
    settings = read_json('settings.json')

    temp_dir = settings['temp_dir']
    symbol_out = settings['symbol_out']

    params_nn_timeframe = params_nn['timeframe']
    params_nn_candle_input_type = params_nn['candle_input_type']

    settings_timeframe = settings['timeframe']
    settings_candle_input_type = settings['candle_input_type']

    if params_nn_timeframe != settings_candle_input_type:
        print(f'ERRO. params_nn_timeframe ({params_nn_timeframe}) != '
              f'settings_candle_input_type ({settings_candle_input_type})')
        exit(-1)

    if params_nn_candle_input_type != settings_timeframe:
        print(f'ERRO. params_nn_candle_input_type ({params_nn_candle_input_type}) != '
              f'settings_timeframe ({settings_timeframe})')
        exit(-1)

    hist = HistMulti(temp_dir, settings_timeframe, symbols_allowed=[symbol_out])

    if symbol_out not in hist.symbols:
        print(f'ERRO. symbol_out ({symbol_out}) not in hist.symbols ({hist.symbols})')
        exit(-1)

    if hist.timeframe != settings_timeframe:
        print(f'o timeframe do diretório {temp_dir} ({hist.timeframe}) é diferente do timeframe especificado '
              f'em settings.json ({settings_timeframe})')
        exit(-1)

    # verifica se o conteúdo de scalers está correto.
    scalers = {}
    with open('scalers.pkl', 'wb') as file:
        pickle.dump(scalers, file)

    symbol_timeframe = f'{symbol_out}_{settings_timeframe}'
    if symbol_timeframe not in scalers:
        print(f'ERRO. symbol_timeframe ({symbol_timeframe}) not in scalers.pkl ({scalers.keys()}).')
        exit(-1)

    n_steps: int = params_nn['n_steps']
    n_hidden_layers: int = params_nn['n_hidden_layers']
