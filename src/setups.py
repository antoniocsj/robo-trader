import os
import pickle
import shutil
import json
import copy

import pandas as pd
from HistMulti import HistMulti
from utils_filesystem import read_json, get_list_sync_files, load_sync_cp_file, update_settings, reset_dir
from utils_symbols import search_symbols
from utils_ops import transform_directory, transform_files, transform_symbols, \
    normalize_directory, normalize_symbols, \
    differentiate_directory, differentiate_files, differentiate_symbols


# As funções 'setup' buscam automatizar parte do trabalho feita na configuração e preparação de um experimento
# com uma rede neural.
# Os experimentos com as redes neurais, geralmente, usam o diretórios csv para formar os conjuntos
# de treinamento e testes.
# symbol_out é o ativo principal no qual serão feitas as negociações de compra e venda.

def check_base_ok():
    """
    O objetivo desta função é verificar se os diretórios temp e csv_s estão prontos para os experimentos a serem
    definidos nas funções 'setup'.
    1) Lê arquivo de settings.json para obter as configurações gerais do experimento.
    2) Reseta o diretório csv, deletando-o e recriando-o novamente.
    3) verifica se a sincronização já está finalizada. caso esteja, verifica se os símbolos presentes no
       diretório csv_s coincidem com a lista de símbolos presente no arquivo de checkpoint final sincronização.
    4) verifica se o símbolo principal {symbol_out} está presente no diretório sincronizado csv_s.
    5) No final, garante que que o diretório csv está resetado e o diretório csv_s com símbolos sincronizados
       corretos.
    :return: True se tudo está OK, False, caso contrário.
    """
    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    csv_s_dir = settings['csv_s_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']

    reset_dir(temp_dir)

    if not os.path.exists(csv_s_dir):
        print(f'o diretório csv_s não foi encontrado.')
        return False

    _sync_files = []
    _sync_files = get_list_sync_files(csv_s_dir)
    if len(_sync_files) == 0:
        print('nenhum arquivo de checkpoint de sincronização foi encontrado.')
        print('não há garantia de que os arquivos foram sinzronizados.')
        return False
    elif len(_sync_files) > 1:
        print('há mais de 1 arquivo de checkpoint de sincronização.')
        print('a sincronização não foi finalizada ainda.')
        return False

    # se chegou até aqui, então len(_sync_files) == 1 e existe o diretório csv_s.
    # verifique se a sincronização já está finalizada. caso esteja, verifique se os símbolos presentes no
    # diretório csv_s coincidem com a lista de símbolos presente no arquivo de checkpoint final sincronização.
    print(_sync_files)
    sync_cp = load_sync_cp_file(csv_s_dir, _sync_files[0])
    print(sync_cp)
    if sync_cp['finished']:
        symbols_to_sync = sync_cp['symbols_to_sync']
        symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)
        if not symbols_names == symbols_to_sync:
            print(f'ERRO. os símbolos encontrados em {csv_s_dir} não coincidem com a lista dos símbolos '
                  f'sincronizados presente no arquivo {_sync_files[0]}')
            return False
    else:
        print('a sincronização não está finalizada ainda.')
        return False

    if symbol_out not in symbols_names:
        print(f'ERRO. o símbolo principal {symbol_out} não está presente no diretório sincronizado csv_s.')
        return False

    # se chegou até aqui, então temos um diretório csv resetado e um diretório csv_s com símbolos sincronizados.
    return True


def csv_delete_first_row(_filepath: str):
    df: pd.DataFrame = pd.read_csv(_filepath, delimiter='\t')
    df.drop(df.index[0], inplace=True)
    df.sort_index(ignore_index=True, inplace=True)
    df.reset_index(drop=True)
    df.to_csv(_filepath, sep='\t', index=False)


def setup_directory_01():
    """
    O diretório temp terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) demais símbolos normalizados.
    :return:
    """
    print('setup_directory_01.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    csv_s_dir = settings['csv_s_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos de csv_s_dir para temp_dir
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{temp_dir}/{symbol}_{timeframe}.csv'
        shutil.copy(_src, _dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(temp_dir)
    update_settings('setup_code', 1)
    update_settings('setup_uses_differentiation', False)


def setup_symbols_01(hist: HistMulti) -> HistMulti:
    """
    O histórico terá os seguintes símbolos:
    -> 1) symbol_out normalizado;
    -> 2) demais símbolos normalizados.
    :return:
    """
    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    symbol_out = settings['symbol_out']

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    _hist = copy.deepcopy(hist)
    normalize_symbols(_hist, scalers)

    return _hist


def setup_directory_02():
    """
    O diretório temp terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) demais símbolos diferenciados e normalizados.
    :return:
    """
    print('setup_directory_02.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    csv_s_dir = settings['csv_s_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos, menos symbol_out, de csv_s_dir para temp_dir
    for symbol in symbols_names:
        if symbol != symbol_out:
            _src = symbols_paths[f'{symbol}_{timeframe}']
            _dst = f'{temp_dir}/{symbol}@D_{timeframe}.csv'
            shutil.copy(_src, _dst)

    # diferenciar os símbolos do diretório csv
    differentiate_directory(temp_dir)

    # copiar symbol_out, de csv_s_dir para temp_dir
    _src = symbols_paths[f'{symbol_out}_{timeframe}']
    _dst = f'{temp_dir}/{symbol_out}_{timeframe}.csv'
    shutil.copy(_src, _dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(temp_dir)

    # como a diferenciação faz os arquivos CSVs (planilhas) perderem a 1a linha, delete a 1a linha do
    # symbol_out também, mas delete do arquivo que está em csv.
    csv_delete_first_row(_dst)

    update_settings('setup_code', 2)
    update_settings('setup_uses_differentiation', True)


def setup_symbols_02(hist: HistMulti) -> HistMulti:
    """
    O histórico terá os seguintes símbolos:
    -> 1) symbol_out normalizado;
    -> 2) demais símbolos diferenciados e normalizados.
    :return:
    """
    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    symbol_out = settings['symbol_out']

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    _hist = copy.deepcopy(hist)

    _hist.remove_symbol(symbol_out)
    _hist.rename_symbols_adding_suffix('@D')
    differentiate_symbols(_hist)

    _hist.copy_symbol(symbol_out, hist)

    _hist.sort_symbols()
    normalize_symbols(_hist, scalers)

    _hist.delete_first_row_symbol(symbol_out)

    return _hist


def setup_directory_03():
    """
    O diretório temp terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) symbol_out diferenciado e normalizado;
    -> 3) demais símbolos diferenciados e normalizados;
    :return:
    """
    print('setup_directory_03.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    csv_s_dir = settings['csv_s_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos de csv_s_dir para temp_dir
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{temp_dir}/{symbol}@D_{timeframe}.csv'
        shutil.copy(_src, _dst)

    # diferenciar os símbolos do diretório csv
    differentiate_directory(temp_dir)

    # copiar symbol_out, de csv_s_dir para temp_dir
    _src = symbols_paths[f'{symbol_out}_{timeframe}']
    _dst = f'{temp_dir}/{symbol_out}_{timeframe}.csv'
    shutil.copy(_src, _dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(temp_dir)

    # como a diferenciação faz os arquivos CSVs (planilhas) perderem a 1a linha, delete a 1a linha do
    # symbol_out também, mas delete do arquivo que está em csv apenas.
    csv_delete_first_row(_dst)

    update_settings('setup_code', 3)
    update_settings('setup_uses_differentiation', True)


def setup_symbols_03(hist: HistMulti) -> HistMulti:
    """
    O histórico terá os seguintes símbolos:
    -> 1) symbol_out normalizado;
    -> 2) symbol_out diferenciado e normalizado;
    -> 3) demais símbolos diferenciados e normalizados;
    :return:
    """
    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    symbol_out = settings['symbol_out']

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    _hist = copy.deepcopy(hist)

    _hist.rename_symbols_adding_suffix('@D')
    differentiate_symbols(_hist)
    _hist.copy_symbol(symbol_out, hist)

    _hist.sort_symbols()
    normalize_symbols(_hist, scalers)
    
    _hist.delete_first_row_symbol(symbol_out)

    return _hist


def setup_directory_04():
    """
    O diretório temp terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) symbol_out diferenciado e normalizado;
    -> 3) demais símbolos normalizados;
    -> 4) demais símbolos diferenciados e normalizados;
    :return:
    """
    print('setup_directory_04.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    csv_s_dir = settings['csv_s_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos, de csv_s_dir para temp_dir, mudando o nome do símbolo (acrescenta D no final).
    # isso é para poder ter dois arquivos do mesmo símbolo. assim, um será diferenciado e normalizados,
    # enquanto que o outro será apenas normalizado.
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{temp_dir}/{symbol}@D_{timeframe}.csv'
        shutil.copy(_src, _dst)

    # diferenciar os símbolos do diretório csv
    differentiate_directory(temp_dir)

    # copiar todos os símbolos, de csv_s_dir para temp_dir
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{temp_dir}/{symbol}_{timeframe}.csv'
        shutil.copy(_src, _dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(temp_dir)

    # como a diferenciação faz os arquivos CSVs (planilhas) perderem a 1a linha,
    # delete a 1a linha do de cada símbolo também, mas delete do arquivo que está em csv apenas.
    for symbol in symbols_names:
        _dst = f'{temp_dir}/{symbol}_{timeframe}.csv'
        csv_delete_first_row(_dst)

    update_settings('setup_code', 4)
    update_settings('setup_uses_differentiation', True)


def setup_symbols_04(hist: HistMulti) -> HistMulti:
    """
    O histórico terá os seguintes símbolos:
    -> 1) symbol_out normalizado;
    -> 2) symbol_out diferenciado e normalizado;
    -> 3) demais símbolos normalizados;
    -> 4) demais símbolos diferenciados e normalizados;
    :return:
    """
    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    symbol_out = settings['symbol_out']

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    _hist = copy.deepcopy(hist)

    _hist.rename_symbols_adding_suffix('@D')
    differentiate_symbols(_hist)

    _hist.sort_symbols()
    normalize_symbols(_hist, scalers)

    _hist.copy_symbols(hist)
    _hist.delete_first_row_symbols(excepting_pattern='@D')

    return _hist


def setup_directory_05():
    """
    O diretório temp terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) symbol_out transformado e normalizado (1 coluna Y: (C-O)*V);
    -> 3) demais símbolos transformados e normalizados (1 coluna Y: (C-O)*V);
    :return:
    """
    print('setup_directory_05.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    csv_s_dir = settings['csv_s_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos, de csv_s_dir para temp_dir, mudando o nome do símbolo (acrescenta @T no final).
    # isso é para poder ter dois arquivos do mesmo símbolo.
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{temp_dir}/{symbol}@T_{timeframe}.csv'
        shutil.copy(_src, _dst)

    transform_directory(temp_dir, '(C-O)*V')

    # copiar symbol_out, de csv_s_dir para temp_dir
    _src = symbols_paths[f'{symbol_out}_{timeframe}']
    _dst = f'{temp_dir}/{symbol_out}_{timeframe}.csv'
    shutil.copy(_src, _dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(temp_dir)
    update_settings('setup_code', 5)
    update_settings('setup_uses_differentiation', False)


def setup_symbols_05(hist: HistMulti) -> HistMulti:
    """
    O histórico terá os seguintes símbolos:
    -> 1) symbol_out normalizado;
    -> 2) symbol_out transformado e normalizado (1 coluna Y: (C-O)*V);
    -> 3) demais símbolos transformados e normalizados (1 coluna Y: (C-O)*V);
    :return:
    """
    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    symbol_out = settings['symbol_out']

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    _hist = copy.deepcopy(hist)

    _hist.rename_symbols_adding_suffix('@T')
    transform_symbols(_hist, '(C-O)*V')
    _hist.copy_symbol(symbol_out, hist)
    _hist.sort_symbols()

    normalize_symbols(_hist, scalers)

    return _hist


def setup_directory_06():
    """
    O diretório temp terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) symbol_out transformado e normalizado (1 coluna Y: (C-O)*V);
    -> 3) demais símbolos normalizados;
    -> 4) demais símbolos transformados e normalizados (1 coluna Y: (C-O)*V);
    :return:
    """
    print('setup_directory_06.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    csv_s_dir = settings['csv_s_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos, de csv_s_dir para temp_dir, mudando o nome do símbolo (acrescenta @T no final).
    # isso é para poder ter dois arquivos do mesmo símbolo.
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{temp_dir}/{symbol}@T_{timeframe}.csv'
        shutil.copy(_src, _dst)

    transform_directory(temp_dir, '(C-O)*V')

    # copiar todos os símbolos, de csv_s_dir para temp_dir
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{temp_dir}/{symbol}_{timeframe}.csv'
        shutil.copy(_src, _dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(temp_dir)
    update_settings('setup_code', 6)
    update_settings('setup_uses_differentiation', False)


def setup_symbols_06(hist: HistMulti) -> HistMulti:
    """
    O histórico terá os seguintes símbolos:
    -> 1) symbol_out normalizado;
    -> 2) symbol_out transformado e normalizado (1 coluna Y: (C-O)*V);
    -> 3) demais símbolos normalizados;
    -> 4) demais símbolos transformados e normalizados (1 coluna Y: (C-O)*V);
    :return:
    """
    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    symbol_out = settings['symbol_out']

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    _hist = copy.deepcopy(hist)

    _hist.rename_symbols_adding_suffix('@T')
    transform_symbols(_hist, '(C-O)*V')

    _hist.copy_symbols(hist)
    _hist.sort_symbols()

    normalize_symbols(_hist, scalers)

    return _hist


def setup_directory_07():
    """
    O diretório temp terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) symbol_out transformado e normalizado (1 coluna Y: (C-O)*V);
    -> 3) symbol_out diferenciado e normalizado;
    -> 4) demais símbolos normalizados;
    -> 5) demais símbolos diferenciados e normalizados;
    -> 6) demais símbolos transformados e normalizados (1 coluna Y: (C-O)*V);
    :return:
    """
    print('setup_directory_07.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    csv_s_dir = settings['csv_s_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos, de csv_s_dir para temp_dir, mudando o nome do símbolo (acrescenta @T no final).
    # isso é para poder ter dois arquivos do mesmo símbolo.
    # símbolos que sofrerão uma transformação.
    _transformed = []
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{temp_dir}/{symbol}@T_{timeframe}.csv'
        shutil.copy(_src, _dst)
        _transformed.append(_dst)

    # transform_directory(temp_dir, '(C-O)*V')
    transform_files(_transformed, temp_dir, '(C-O)*V')

    # copiar todos os símbolos, de csv_s_dir para temp_dir
    # símbolos normais, apenas serão normalizados depois.
    _normals = []
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{temp_dir}/{symbol}_{timeframe}.csv'
        shutil.copy(_src, _dst)
        _normals.append(_dst)

    # adicionando os símbolos que sofrerão a operação diferenciação.
    _differentiated = []
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{temp_dir}/{symbol}@D_{timeframe}.csv'
        shutil.copy(_src, _dst)
        _differentiated.append(_dst)

    differentiate_files(_differentiated, temp_dir)

    # normaliza todos os symbolos de csv.
    normalize_directory(temp_dir)

    # como a diferenciação faz os arquivos CSVs (planilhas) perderem a 1a linha, delete também  a 1a linha dos
    # outros símbolos que não sofreram a operação diferenciação.
    _list = _transformed + _normals
    for _filepath in _list:
        csv_delete_first_row(_filepath)

    update_settings('setup_code', 7)
    update_settings('setup_uses_differentiation', True)


def setup_symbols_07(hist: HistMulti) -> HistMulti:
    """
    O histórico terá os seguintes símbolos:
    -> 1) symbol_out normalizado;
    -> 2) symbol_out transformado e normalizado (1 coluna Y: (C-O)*V);
    -> 3) symbol_out diferenciado e normalizado;
    -> 4) demais símbolos normalizados;
    -> 5) demais símbolos diferenciados e normalizados;
    -> 6) demais símbolos transformados e normalizados (1 coluna Y: (C-O)*V);
    :return:
    """
    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    symbol_out = settings['symbol_out']

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    _hist = copy.deepcopy(hist)

    _hist.rename_symbols_adding_suffix('@T')
    _transformed = _hist.symbols[:]
    transform_symbols(_hist, '(C-O)*V')

    _normals = hist.symbols[:]
    _hist.copy_symbols(hist)

    _differentiated = []
    for symbol_name_src in hist.symbols:
        symbol_name_dst = f'{symbol_name_src}@D'
        _hist.copy_symbol(symbol_name_src, hist, symbol_name_dst)
        _differentiated.append(symbol_name_dst)

    differentiate_symbols(_hist, _differentiated)

    _hist.sort_symbols()
    normalize_symbols(_hist, scalers)

    _list = _transformed + _normals
    for s in _list:
        _hist.delete_first_row_symbol(s)

    return _hist


def setup_directory_08():
    """
    O diretório temp terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) symbol_out diferenciado, transformado e normalizado (1 coluna Y: (C-O)*V);
    -> 3) demais símbolos diferenciados, transformados e normalizados (1 coluna Y: (C-O)*V);
    :return:
    """
    print('setup_directory_08.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    csv_s_dir = settings['csv_s_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos, de csv_s_dir para temp_dir, mudando o nome do símbolo (acrescenta @ no final).
    # isso é para poder ter dois arquivos do mesmo símbolo.
    # símbolos que sofrerão uma transformação.
    _transformed = []
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{temp_dir}/{symbol}@DT_{timeframe}.csv'
        shutil.copy(_src, _dst)
        _transformed.append(_dst)

    differentiate_directory(temp_dir)
    transform_directory(temp_dir, '(C-O)*V')

    # copiar symbol_out, de csv_s_dir para temp_dir
    _src = symbols_paths[f'{symbol_out}_{timeframe}']
    _dst = f'{temp_dir}/{symbol_out}_{timeframe}.csv'
    shutil.copy(_src, _dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(temp_dir)

    # como a diferenciação faz os arquivos CSVs (planilhas) perderem a 1a linha, delete a 1a linha do
    # symbol_out também, mas delete do arquivo que está em csv.
    csv_delete_first_row(_dst)
    update_settings('setup_code', 8)
    update_settings('setup_uses_differentiation', True)


def setup_symbols_08(hist: HistMulti) -> HistMulti:
    """
    O histórico terá os seguintes símbolos:
    -> 1) symbol_out normalizado;
    -> 2) symbol_out diferenciado, transformado e normalizado (1 coluna Y: (C-O)*V);
    -> 3) demais símbolos diferenciados, transformados e normalizados (1 coluna Y: (C-O)*V);
    :return:
    """
    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    symbol_out = settings['symbol_out']

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    _hist = copy.deepcopy(hist)

    _hist.rename_symbols_adding_suffix('@DT')
    differentiate_symbols(_hist)
    transform_symbols(_hist, '(C-O)*V')

    _hist.copy_symbol(symbol_out, hist)

    _hist.sort_symbols()
    normalize_symbols(_hist, scalers)

    _hist.delete_first_row_symbol(symbol_out)

    return _hist


def setup_directory_09():
    """
    O diretório temp terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) symbol_out transformado, diferenciado e normalizado (1 coluna Y: (C-O)*V);
    -> 3) demais símbolos transformados, diferenciados e normalizados (1 coluna Y: (C-O)*V);
    :return:
    """
    print('setup_directory_09.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    settings = read_json('settings.json')
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    csv_s_dir = settings['csv_s_dir']
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos, de csv_s_dir para temp_dir, mudando o nome do símbolo (acrescenta @ no final).
    # isso é para poder ter dois arquivos do mesmo símbolo.
    # símbolos que sofrerão uma transformação.
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{temp_dir}/{symbol}@TD_{timeframe}.csv'
        shutil.copy(_src, _dst)

    transform_directory(temp_dir, '(C-O)*V')
    differentiate_directory(temp_dir)

    # copiar symbol_out, de csv_s_dir para temp_dir
    _src = symbols_paths[f'{symbol_out}_{timeframe}']
    _dst = f'{temp_dir}/{symbol_out}_{timeframe}.csv'
    shutil.copy(_src, _dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(temp_dir)

    # como a diferenciação faz os arquivos CSVs (planilhas) perderem a 1a linha, delete a 1a linha do
    # symbol_out também, mas delete do arquivo que está em csv.
    csv_delete_first_row(_dst)
    update_settings('setup_code', 9)
    update_settings('setup_uses_differentiation', True)


def setup_symbols_09(hist: HistMulti) -> HistMulti:
    """
    O histórico terá os seguintes símbolos:
    -> 1) symbol_out normalizado;
    -> 2) symbol_out transformado, diferenciado e normalizado (1 coluna Y: (C-O)*V);
    -> 3) demais símbolos transformados, diferenciados e normalizados (1 coluna Y: (C-O)*V);
    :return:
    """
    with open('settings.json', 'r') as file:
        settings = json.load(file)
    print(f'settings.json: {settings}')

    symbol_out = settings['symbol_out']

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    _hist = copy.deepcopy(hist)

    _hist.rename_symbols_adding_suffix('@TD')
    transform_symbols(_hist, '(C-O)*V')
    differentiate_symbols(_hist)

    _hist.copy_symbol(symbol_out, hist)

    _hist.sort_symbols()
    normalize_symbols(_hist, scalers)

    _hist.delete_first_row_symbol(symbol_out)

    return _hist


def apply_setup_symbols(hist: HistMulti, code: int) -> HistMulti:
    if code == 1:
        return setup_symbols_01(hist)
    elif code == 2:
        return setup_symbols_02(hist)
    elif code == 3:
        return setup_symbols_03(hist)
    elif code == 4:
        return setup_symbols_04(hist)
    elif code == 5:
        return setup_symbols_05(hist)
    elif code == 6:
        return setup_symbols_06(hist)
    elif code == 7:
        return setup_symbols_07(hist)
    elif code == 8:
        return setup_symbols_08(hist)
    elif code == 9:
        return setup_symbols_09(hist)
    else:
        print(F'ERRO. setup_code ({code}) inválido.')
        exit(-1)


if __name__ == '__main__':
    setup_directory_09()
