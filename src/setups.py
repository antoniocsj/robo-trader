import os
import shutil
import json
from utils import get_list_sync_files


# As funções 'setup' buscam automatizar parte do trabalho feita na configuração e preparação de um experimento
# com uma rede neural.

def setup_01():
    """
    1) Lê arquivo de setup.json para obter as configurações gerais do experimento, como o 'symbol_out'.
    2) Reseta o diretório csv, deletando-o e recriando-o novamente.
    3) Obtém a lista de todos os arquivos contidos no diretório dos CSVs sincronizados 'csv_s'
    :return:
    """
    setup = {}
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


if __name__ == '__main__':
    setup_01()
