import os
import shutil
import json

import pandas as pd
from utils_filesystem import get_list_sync_files, load_sync_cp_file
from utils_symbols import search_symbols
from utils_ops import differentiate_directory, normalize_directory, transform_directory, differentiate_files, \
    transform_files


# As funções 'setup' buscam automatizar parte do trabalho feita na configuração e preparação de um experimento
# com uma rede neural.
# Os experimentos com as redes neurais, geralmente, usam o diretórios csv para formar os conjuntos
# de treinamento e testes.
# symbol_out é o ativo principal no qual serão feitas as negociações de compra e venda.

def check_base_ok():
    """
    O objetivo desta função é verificar se os diretórios csv e csv_s estão prontos para os experimentos a serem
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
        setup = json.load(file)
    print(f'settings.json: {setup}')

    csv_dir = setup['csv_dir']
    csv_s_dir = setup['csv_s_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']

    if os.path.exists(csv_dir):
        print(f'o diretório {csv_dir} já existe. deletando todo seu conteúdo.')
        shutil.rmtree(csv_dir)
        os.mkdir(csv_dir)
        _filename = f'{csv_dir}/.directory'
        _f = open(_filename, 'x')  # para manter o diretório no git
        _f.close()
    else:
        print(f'o diretório {csv_dir} não existe. criando-o.')
        os.mkdir(csv_dir)
        _filename = f'{csv_dir}/.directory'
        _f = open(_filename, 'x')  # para manter o diretório no git
        _f.close()

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


def setup_01():
    """
    O diretório csv terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) demais símbolos normalizados.
    :return:
    """
    print('setup_01.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        setup = json.load(file)
    print(f'settings.json: {setup}')

    csv_dir = setup['csv_dir']
    csv_s_dir = setup['csv_s_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos de csv_s_dir para csv_dir
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{csv_dir}/{symbol}_{timeframe}.csv'
        shutil.copy(_src, _dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(csv_dir)


def setup_02():
    """
    O diretório csv terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) demais símbolos diferenciados e normalizados.
    :return:
    """
    print('setup_02.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        setup = json.load(file)
    print(f'settings.json: {setup}')

    csv_dir = setup['csv_dir']
    csv_s_dir = setup['csv_s_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos, menos symbol_out, de csv_s_dir para csv_dir
    for symbol in symbols_names:
        if symbol != symbol_out:
            _src = symbols_paths[f'{symbol}_{timeframe}']
            _dst = f'{csv_dir}/{symbol}@D_{timeframe}.csv'
            shutil.copy(_src, _dst)

    # diferenciar os símbolos do diretório csv
    differentiate_directory(csv_dir)

    # copiar symbol_out, de csv_s_dir para csv_dir
    _src = symbols_paths[f'{symbol_out}_{timeframe}']
    _dst = f'{csv_dir}/{symbol_out}_{timeframe}.csv'
    shutil.copy(_src, _dst)

    # como a diferenciação faz os arquivos CSVs (planilhas) perderem a 1a linha, delete a 1a linha do
    # symbol_out também, mas delete do arquivo que está em csv.
    csv_delete_first_row(_dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(csv_dir)


def setup_03():
    """
    O diretório csv terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) symbol_out diferenciado e normalizado;
    -> 3) demais símbolos diferenciados e normalizados;
    :return:
    """
    print('setup_03.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        setup = json.load(file)
    print(f'settings.json: {setup}')

    csv_dir = setup['csv_dir']
    csv_s_dir = setup['csv_s_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos de csv_s_dir para csv_dir
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{csv_dir}/{symbol}@D_{timeframe}.csv'
        shutil.copy(_src, _dst)

    # diferenciar os símbolos do diretório csv
    differentiate_directory(csv_dir)

    # copiar symbol_out, de csv_s_dir para csv_dir
    _src = symbols_paths[f'{symbol_out}_{timeframe}']
    _dst = f'{csv_dir}/{symbol_out}_{timeframe}.csv'
    shutil.copy(_src, _dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(csv_dir)

    # como a diferenciação faz os arquivos CSVs (planilhas) perderem a 1a linha, delete a 1a linha do
    # symbol_out também, mas delete do arquivo que está em csv apenas.
    csv_delete_first_row(_dst)


def setup_04():
    """
    O diretório csv terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) symbol_out diferenciado e normalizado;
    -> 3) demais símbolos normalizados;
    -> 4) demais símbolos diferenciados e normalizados;
    :return:
    """
    print('setup_04.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        setup = json.load(file)
    print(f'settings.json: {setup}')

    csv_dir = setup['csv_dir']
    csv_s_dir = setup['csv_s_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos, de csv_s_dir para csv_dir, mudando o nome do símbolo (acrescenta D no final).
    # isso é para poder ter dois arquivos do mesmo símbolo. assim, um será diferenciado e normalizados,
    # enquanto que o outro será apenas normalizado.
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{csv_dir}/{symbol}@D_{timeframe}.csv'
        shutil.copy(_src, _dst)

    # diferenciar os símbolos do diretório csv
    differentiate_directory(csv_dir)

    # copiar todos os símbolos, de csv_s_dir para csv_dir
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{csv_dir}/{symbol}_{timeframe}.csv'
        shutil.copy(_src, _dst)
        # como a diferenciação faz os arquivos CSVs (planilhas) perderem a 1a linha,
        # delete a 1a linha do de cada símbolo também, mas delete do arquivo que está em csv apenas.
        csv_delete_first_row(_dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(csv_dir)


def setup_05():
    """
    O diretório csv terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) symbol_out transformado e normalizado (1 coluna Y: (C-O)*V);
    -> 3) demais símbolos transformados e normalizado (1 coluna Y: (C-O)*V);
    :return:
    """
    print('setup_05.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        setup = json.load(file)
    print(f'settings.json: {setup}')

    csv_dir = setup['csv_dir']
    csv_s_dir = setup['csv_s_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos, de csv_s_dir para csv_dir, mudando o nome do símbolo (acrescenta @T no final).
    # isso é para poder ter dois arquivos do mesmo símbolo.
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{csv_dir}/{symbol}@T_{timeframe}.csv'
        shutil.copy(_src, _dst)

    transform_directory(csv_dir, '(C-O)*V')

    # copiar symbol_out, de csv_s_dir para csv_dir
    _src = symbols_paths[f'{symbol_out}_{timeframe}']
    _dst = f'{csv_dir}/{symbol_out}_{timeframe}.csv'
    shutil.copy(_src, _dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(csv_dir)


def setup_06():
    """
    O diretório csv terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) symbol_out transformado e normalizado (1 coluna Y: (C-O)*V);
    -> 3) demais símbolos normalizados;
    -> 4) demais símbolos transformados e normalizado (1 coluna Y: (C-O)*V);
    :return:
    """
    print('setup_06.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        setup = json.load(file)
    print(f'settings.json: {setup}')

    csv_dir = setup['csv_dir']
    csv_s_dir = setup['csv_s_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos, de csv_s_dir para csv_dir, mudando o nome do símbolo (acrescenta @T no final).
    # isso é para poder ter dois arquivos do mesmo símbolo.
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{csv_dir}/{symbol}@T_{timeframe}.csv'
        shutil.copy(_src, _dst)

    transform_directory(csv_dir, '(C-O)*V')

    # copiar todos os símbolos, de csv_s_dir para csv_dir
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{csv_dir}/{symbol}_{timeframe}.csv'
        shutil.copy(_src, _dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(csv_dir)


def setup_07():
    """
    O diretório csv terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) symbol_out transformado e normalizado (1 coluna Y: (C-O)*V);
    -> 3) symbol_out diferenciado e normalizado;
    -> 4) demais símbolos normalizados;
    -> 5) demais símbolos diferenciados e normalizados;
    -> 4) demais símbolos transformados e normalizado (1 coluna Y: (C-O)*V);
    :return:
    """
    print('setup_07.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        setup = json.load(file)
    print(f'settings.json: {setup}')

    csv_dir = setup['csv_dir']
    csv_s_dir = setup['csv_s_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos, de csv_s_dir para csv_dir, mudando o nome do símbolo (acrescenta @T no final).
    # isso é para poder ter dois arquivos do mesmo símbolo.
    # símbolos que sofrerão uma transformação.
    _transformed = []
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{csv_dir}/{symbol}@T_{timeframe}.csv'
        shutil.copy(_src, _dst)
        _transformed.append(_dst)

    # transform_directory(csv_dir, '(C-O)*V')
    transform_files(_transformed, csv_dir, '(C-O)*V')

    # copiar todos os símbolos, de csv_s_dir para csv_dir
    # símbolos normais, apenas serão normalizados depois.
    _normals = []
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{csv_dir}/{symbol}_{timeframe}.csv'
        shutil.copy(_src, _dst)
        _normals.append(_dst)

    # adicionando os símbolos que sofrerão a operação diferenciação.
    _differentiated = []
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{csv_dir}/{symbol}@D_{timeframe}.csv'
        shutil.copy(_src, _dst)
        _differentiated.append(_dst)

    differentiate_files(_differentiated, csv_dir)

    # normaliza todos os symbolos de csv.
    normalize_directory(csv_dir)

    # como a diferenciação faz os arquivos CSVs (planilhas) perderem a 1a linha, delete também  a 1a linha dos
    # outros símbolos que não sofreram a operação diferenciação.
    _list = _transformed + _normals
    for _filepath in _list:
        csv_delete_first_row(_filepath)


def setup_08():
    """
    O diretório csv terá os seguintes símbolos (arquivos CSVs):
    -> 1) symbol_out normalizado;
    -> 2) symbol_out diferenciado, transformado e normalizado (1 coluna Y: C*V);
    -> 4) demais símbolos diferenciados, transformados e normalizados (1 coluna Y: C*V);
    :return:
    """
    print('setup_08.')
    if not check_base_ok():
        print('abortando setup.')
        exit(-1)

    with open('settings.json', 'r') as file:
        setup = json.load(file)
    print(f'settings.json: {setup}')

    csv_dir = setup['csv_dir']
    csv_s_dir = setup['csv_s_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']
    symbols_names, symbols_paths = search_symbols(csv_s_dir, timeframe)

    # copiar todos os símbolos, de csv_s_dir para csv_dir, mudando o nome do símbolo (acrescenta @ no final).
    # isso é para poder ter dois arquivos do mesmo símbolo.
    # símbolos que sofrerão uma transformação.
    _transformed = []
    for symbol in symbols_names:
        _src = symbols_paths[f'{symbol}_{timeframe}']
        _dst = f'{csv_dir}/{symbol}@DT_{timeframe}.csv'
        shutil.copy(_src, _dst)
        _transformed.append(_dst)

    differentiate_directory(csv_dir)
    transform_directory(csv_dir, 'C*V')

    # copiar symbol_out, de csv_s_dir para csv_dir
    _src = symbols_paths[f'{symbol_out}_{timeframe}']
    _dst = f'{csv_dir}/{symbol_out}_{timeframe}.csv'
    shutil.copy(_src, _dst)

    # normaliza todos os symbolos de csv.
    normalize_directory(csv_dir)

    # como a diferenciação faz os arquivos CSVs (planilhas) perderem a 1a linha, delete a 1a linha do
    # symbol_out também, mas delete do arquivo que está em csv.
    csv_delete_first_row(_dst)


if __name__ == '__main__':
    setup_01()
