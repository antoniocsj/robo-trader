import os
import pickle
from src.utils.utils_filesystem import read_json
from src.HistMulti import HistMulti


def initial_compliance_checks(working_dir: str):
    """
    Realiza as verificações iniciais de conformidade:
    -> o conteúdo de temp, scalers.pkl deve estar em conformidade com params_rs_search.json e settings.json
    :return:
    """
    print('Realizando as verificações iniciais de conformidade.')

    params_rs_search_filepath = os.path.join(working_dir, 'params_rs_search.json')
    params_rs_search = read_json(params_rs_search_filepath)

    settings_filepath = os.path.join(working_dir, 'settings.json')
    settings = read_json(settings_filepath)

    temp_dir = os.path.join(working_dir, settings['temp_dir'])
    symbol_out = settings['symbol_out']

    params_tf = params_rs_search['timeframe']
    params_cit = params_rs_search['candle_input_type']

    settings_tf = settings['timeframe']
    settings_cit = settings['candle_input_type']

    if params_tf != settings_tf:
        print(f'ERRO. params_rs_search["timeframe"] ({params_tf}) != settings["timeframe"] ({settings_tf})')
        exit(-1)

    if params_cit != settings_cit:
        print(f'ERRO. params_rs_search["candle_input_type"] ({params_cit}) != settings["candle_input_type"] ({settings_cit})')
        exit(-1)

    hist = HistMulti(temp_dir, settings_tf, symbols_allowed=[symbol_out])

    if symbol_out not in hist.symbols:
        print(f'ERRO. symbol_out ({symbol_out}) not in hist.symbols ({hist.symbols})')
        exit(-1)

    if hist.timeframe != settings_tf:
        print(f'o timeframe do diretório {temp_dir} ({hist.timeframe}) é diferente do timeframe especificado '
              f'em settings.json ({settings_tf})')
        exit(-1)

    del hist

    # caso o histórico tenha sofrido normalização, verifica se o conteúdo de scalers está correto.
    if settings['setup_code'] > 0:
        scalers_filepath = os.path.join(temp_dir, 'scalers.pkl')

        if os.path.exists(scalers_filepath):
            with open(scalers_filepath, 'rb') as file:
                scalers: dict = pickle.load(file)
        else:
            print(f'ERRO. O arquivo {scalers_filepath} não foi encontrado.')
            print('talvez você precise rodar o script setups.py')
            exit(-1)

        symbol_timeframe = f'{symbol_out}_{settings_tf}'
        if symbol_timeframe not in scalers:
            print(f'ERRO. symbol_timeframe ({symbol_timeframe}) not in scalers.pkl ({scalers.keys()}).')
            exit(-1)

    print('Verificações Iniciais de Conformidade: OK.')
