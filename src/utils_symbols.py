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


def calc_n_inputs(directory: str, tipo_vela: str, timeframe: str):
    """
    Faz uma varredura no diretório e retorna o número de colunas (além da data e horário) que há em cada arquivos
    CSV e também retorna o número de símbolos/arquivos CSVs
    :param directory:
    :param tipo_vela:
    :param timeframe:
    :return:
    """
    count = 0
    symbols_names, symbols_paths = search_symbols(directory, timeframe)
    for s in symbols_names:
        if s.endswith('@T') or s.endswith('@DT'):
            count += 1
        else:
            count += len(tipo_vela)
    return count, len(symbols_names)
