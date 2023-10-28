from utils_filesystem import read_json
from HistMulti import HistMulti


def initial_compliance_checks():
    """
    Realiza as verificações iniciais de conformidade:
    -> conteúdo de temp, scalers.pkl em conformidade com settings.json e params_nn.json
    :return:
    """
    everything_is_ok = True

    params_nn = read_json('params_nn.json')
    settings = read_json('settings.json')

    temp_dir = settings['temp_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    candle_input_type = settings['candle_input_type']
    candle_output_type = settings['candle_output_type']
    hist = HistMulti(temp_dir, timeframe, symbols_allowed=[symbol_out])
    datetime_start = hist.arr[symbol_out][timeframe][0][0]
    datetime_end = hist.arr[symbol_out][timeframe][-1][0]

    if hist.timeframe != timeframe:
        print(f'o timeframe do diretório {temp_dir} ({hist.timeframe}) é diferente do timeframe especificado '
              f'em settings.json ({timeframe})')
        exit(-1)

    n_steps: int = params_nn['n_steps']
    n_hidden_layers: int = params_nn['n_hidden_layers']