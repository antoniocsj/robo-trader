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
    symbols_currencies_majors = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY']
    symbols_currencies_crosses = ['AUDJPY', 'AUDCAD', 'AUDCHF', 'EURAUD', 'EURCHF', 'EURGBP', 'GBPAUD', 'GBPCHF',
                                  'CADCHF', 'CADJPY', 'CHFJPY', 'EURCAD', 'EURJPY', 'GBPCAD', 'GBPJPY']
    symbols_commodities_gold = ['XAUUSD', 'XAUAUD', 'XAUCHF', 'XAUEUR', 'XAUGBP', 'XAUJPY']
    symbols_commodities_silver = ['XAGAUD', 'XAGEUR', 'XAGUSD']
    symbols_indices_majors = ['AUS200', 'EUSTX50', 'FRA40', 'GER40', 'JPN225', 'NAS100', 'UK100', 'US30', 'US500']
    symbols_indices_minors = ['CA60', 'SWI20', 'US2000']
    symbols_indices_currency = ['EURX', 'JPYX', 'USDX']

    if not categories:
        all_symbols = symbols_currencies_majors + symbols_currencies_crosses
        all_symbols += symbols_commodities_gold + symbols_commodities_silver
        all_symbols += symbols_indices_majors + symbols_indices_minors + symbols_indices_currency
        return sorted(all_symbols)
    else:
        if categories == 'currencies_majors':
            all_symbols = sorted(symbols_currencies_majors)
            return all_symbols
        elif categories == 'currencies_crosses':
            all_symbols = sorted(symbols_currencies_crosses)
            return all_symbols
        elif categories == 'commodities_gold':
            all_symbols = sorted(symbols_commodities_gold)
            return all_symbols
        elif categories == 'commodities_silver':
            all_symbols = sorted(symbols_commodities_silver)
            return all_symbols
        elif categories == 'indices_majors':
            all_symbols = sorted(symbols_indices_majors)
            return all_symbols
        elif categories == 'indices_minors':
            all_symbols = sorted(symbols_indices_minors)
            return all_symbols
        elif categories == 'indices_currency':
            all_symbols = sorted(symbols_indices_currency)
            return all_symbols
        else:
            print(f'ERRO. get_symbols(). categoria de símbolos não suportada: {categories}')
            exit(-1)


if __name__ == '__main__':

    symbols = get_symbols()
    print(symbols)
