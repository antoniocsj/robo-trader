import os
import shutil
import json
from utils import get_list_sync_files, load_sync_cp_file, search_symbols


# As funções 'setup' buscam automatizar parte do trabalho feita na configuração e preparação de um experimento
# com uma rede neural.

def setup_01():
    """
    1) Lê arquivo de setup.json para obter as configurações gerais do experimento, como o 'symbol_out'.
    2) Reseta o diretório csv, deletando-o e recriando-o novamente.
    3) Obtém a lista de todos os arquivos contidos no diretório dos CSVs sincronizados 'csv_s'
    :return:
    """
    with open('setup.json', 'r') as file:
        setup = json.load(file)
    print(f'setup.json: {setup}')

    csv_dir = setup['csv_dir']
    csv_s_dir = setup['csv_s_dir']

    if os.path.exists(csv_dir):
        print(f'o diretório {csv_dir} já existe. deletando todo seu conteúdo.')
        shutil.rmtree(csv_dir)
        os.mkdir(csv_dir)
        _filename = f'{csv_dir}/.directory'
        _f = open(_filename, 'x')  # para manter o diretório no git
        _f.close()

    _sync_files = []
    _sync_files = get_list_sync_files()
    if len(_sync_files) == 0:
        print('nenhum arquivo de checkpoint de sincronização foi encontrado.')
        print('não há garantia de que os arquivos foram sinzronizados.')
        exit(-1)
    elif len(_sync_files) > 1:
        print('há mais de 1 arquivo de checkpoint de sincronização.')
        print('a sincronização não foi finalizada ainda.')
        exit(-1)

    if not os.path.exists(csv_s_dir):
        print(f'o diretório csv_s não foi encontrado.')
        exit(-1)

    # se chegou até aqui, então len(_sync_files) == 1 e existe o diretório csv_s.
    # verifique se a sincronização já está finalizada. caso esteja, verifique se os símbolos presentes no
    # diretório csv_s coincidem com a lista de símbolos presente no arquivo de checkpoint final sincronização.
    print(_sync_files)
    sync_cp = load_sync_cp_file(_sync_files[0])
    print(sync_cp)
    if sync_cp['finished']:
        symbols_to_sync = sync_cp['symbols_to_sync']
        symbols_found = search_symbols(csv_s_dir)
        if not symbols_found == symbols_to_sync:
            print(f'ERRO. os símbolos encontrados em {csv_s_dir} não coincidem com a lista dos símbolos '
                  f'sincronizados presente no arquivo {_sync_files[0]}')
            exit(-1)
    else:
        print('a sincronização não está finalizada ainda.')
        exit(-1)

    # se chegou até aqui, então temos um diretório csv resetado e um diretório csv_s sincronizado.


if __name__ == '__main__':
    setup_01()
