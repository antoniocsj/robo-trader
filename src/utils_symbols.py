import os


def search_symbols_in_directory(directory: str, timeframe: str) -> list[str]:
    """
    Procurando pelos símbolos presentes num diretório contendo arquivos csv.
    Todos os arquivos devem ser do timeframe especificado.
    :return: lista dos símbolos
    """
    # passe por todos os arquivos csv e descubra o symbol e timeframe
    symbols = []
    all_files = os.listdir(directory)

    for filename in all_files:
        if filename.endswith('.csv'):
            _symbol = filename.split('_')[0]
            _timeframe = filename.split('_')[1]
            if _timeframe.endswith('.csv'):
                _timeframe = _timeframe.replace('.csv', '')

            if _symbol not in symbols:
                symbols.append(_symbol)
            else:
                print(f'erro. o símbolo {_symbol} aparece repetido no mesmo diretório')
                exit(-1)

            if _timeframe != timeframe:
                print(f'ERRO. timeframe {_timeframe} diferente do especificado {timeframe} foi '
                      f'encontrado no diretório {directory}')
                exit(-1)

    symbols = sorted(symbols)
    return symbols


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


def search_symbols(directory: str, timeframe: str):
    """
    Procurando pelos símbolos nos nomes dos arquivos csv.
    Considera erro encontrar símbolos repetidos ou mais de 1 timeframe.
    :return: lista dos símbolos ordenada alfabeticamente e um dicionário contendo os caminhos relativos
             para os arquivos CSVs correspondentes.
    """
    # passe por todos os arquivos csv e descubra o symbol e timeframe
    if not os.path.exists(directory):
        print(f'ERRO. o diretório {directory} não existe.')
        exit(-1)

    symbols_names = []
    timeframes = set()
    symbols_paths = {}

    all_files = os.listdir(directory)
    for filename in all_files:
        if filename.endswith('.csv'):
            _symbol = filename.split('_')[0]
            _timeframe = filename.split('_')[1]
            if _timeframe.endswith('.csv'):
                _timeframe = _timeframe.replace('.csv', '')

            if _symbol not in symbols_names:
                symbols_names.append(_symbol)
                _filepath = f'{directory}/{filename}'
                symbols_paths[f'{_symbol}_{_timeframe}'] = _filepath
                timeframes.add(_timeframe)
            else:
                print(f'ERRO. o símbolo {_symbol} aparece repetido no mesmo diretório')
                exit(-1)

    if len(timeframes) == 0:
        print(f'ERRO. Não foram encontrados símbolos válidos no diretório {directory}')
        exit(-1)

    if len(timeframes) > 1:
        print(f'ERRO. Há mais de 1 timeframe no diretório {directory}')
        exit(-1)

    if list(timeframes)[0] != timeframe:
        print(f'ERRO. O diretório {directory} não possui símbolos com timeframe {timeframe}')
        exit(-1)

    symbols_names = sorted(symbols_names)
    return symbols_names, symbols_paths


def calc_n_features(directory: str, candle_input_type: str, timeframe: str):
    """
    Faz uma varredura no diretório e retorna o número de colunas (além da data e horário) que há em cada arquivos
    CSV e também retorna o número de símbolos/arquivos CSVs.
    número de colunas = n_features.
    :param directory:
    :param candle_input_type:
    :param timeframe:
    :return: n_features (n_cols), número de símbolos (CSVs)
    """
    count = 0
    symbols_names, symbols_paths = search_symbols(directory, timeframe)
    for s in symbols_names:
        if s.endswith('@T') or s.endswith('@DT') or s.endswith('@TD'):
            count += 1
        else:
            count += len(candle_input_type)
    return count, len(symbols_names)


def get_symbols(categories: str = None) -> list[str]:
    currencies_majors = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY']
    currencies_crosses = ['AUDJPY', 'AUDCAD', 'AUDCHF', 'EURAUD', 'EURCHF', 'EURGBP', 'GBPAUD', 'GBPCHF',
                          'CADCHF', 'CADJPY', 'CHFJPY', 'EURCAD', 'EURJPY', 'GBPCAD', 'GBPJPY']
    commodities_gold = ['XAUUSD', 'XAUAUD', 'XAUCHF', 'XAUEUR', 'XAUGBP', 'XAUJPY']
    commodities_silver = ['XAGAUD', 'XAGEUR', 'XAGUSD']
    indices_majors = ['AUS200', 'EUSTX50', 'FRA40', 'GER40', 'JPN225', 'NAS100', 'UK100', 'US30', 'US500']
    indices_minors = ['CA60', 'SWI20', 'US2000']
    indices_currency = ['EURX', 'JPYX', 'USDX']

    symbols_dict = {
        'currencies_majors': currencies_majors,
        'currencies_crosses': currencies_crosses,
        'commodities_gold': commodities_gold,
        'commodities_silver': commodities_silver,
        'indices_majors': indices_majors,
        'indices_minors': indices_minors,
        'indices_currency': indices_currency
    }

    if not categories:
        symbols = currencies_majors + currencies_crosses
        symbols += commodities_gold + commodities_silver
        symbols += indices_majors + indices_minors + indices_currency
        return sorted(symbols)

    else:
        categories = categories.lower()
        keys = categories.split('+')

        symbols = []
        for key in keys:
            symbols += symbols_dict[key]

        return sorted(symbols)


if __name__ == '__main__':
    s = get_symbols()
    print(s)

    s = get_symbols('currencies_majors')
    print(s)

    s = get_symbols('currencies_crosses')
    print(s)

    s = get_symbols('currencies_majors+currencies_crosses')
    print(s)

    s = get_symbols('commodities_gold+indices_majors')
    print(s)
