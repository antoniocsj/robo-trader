import math
import multiprocessing as mp
import shutil

from utils_filesystem import get_list_sync_files, read_json, copy_files, reset_dir
from utils_symbols import search_symbols_in_directory, get_symbols
from DirectorySynchronizationMultiProc import DirectorySynchronization, make_backup, choose_n_procs_start
from utils_sync import *


# synchronizer


def synchronize() -> bool:
    setup = read_json('settings.json')
    temp_dir = setup['temp_dir']
    csv_o_dir = setup['csv_o_dir']
    csv_s_dir = setup['csv_s_dir']
    timeframe = setup['timeframe']

    # se o diretório temp não existe, então crie-o.
    if not os.path.exists(temp_dir):
        print('o diretório temp não existe. criando-o.')
        os.mkdir(temp_dir)
        _filename = f'{temp_dir}/.directory'
        _f = open(_filename, 'x')  # para manter o diretório no git
        _f.close()

    symbols = search_symbols_in_directory(temp_dir, timeframe)
    _len_symbols = len(symbols)

    if _len_symbols == 0:
        print('Não há arquivos CSVs para serem sincronizados.')
        return True
    elif _len_symbols == 1:
        print('Há apenas 1 arquivo CSV. Portanto, considera-se que o arquivo já está sincronizado.')
        make_backup(temp_dir, csv_s_dir)
        return True

    symbols_to_sync_per_proc = []
    list_sync_files = get_list_sync_files('.')

    if len(list_sync_files) == 0:
        print('iniciando a sincronização dos arquivos csv pela PRIMEIRA vez.')

        n_procs = choose_n_procs_start(_len_symbols)
        pool = mp.Pool(n_procs)

        for i in range(n_procs):
            symbols_to_sync_per_proc.append([])

        for i in range(len(symbols)):
            i_proc = i % n_procs
            symbols_to_sync_per_proc[i_proc].append(symbols[i])

        dir_sync_l: list[DirectorySynchronization] = []
        for i in range(n_procs):
            dir_sync = DirectorySynchronization(temp_dir, timeframe, i, symbols_to_sync_per_proc[i])
            dir_sync_l.append(dir_sync)

        for i in range(n_procs):
            pool.apply_async(dir_sync_l[i].synchronize_directory, args=(i,))
        pool.close()
        pool.join()

    else:
        print('pode haver sincronização em andamento.')
        print(f'checkpoints: {list_sync_files}')
        # verifique se todos os checkpoints indicam sincronização finalizada
        _results_set = set()
        for i in range(len(list_sync_files)):
            _sync_status = get_sync_status(list_sync_files[i])
            _results_set.add(_sync_status)

        if len(_results_set) == 1 and list(_results_set)[0] is True:
            print('TODOS os checkpoints indicam que suas sincronizações estão FINALIZADAS')
            # assume-se que sempre há um número de processos/num_sync_cp_files que seja uma potência de 2
            # pois será feita uma fusão de cada 2 conjuntos de símbolos de modo que na próxima sincronização
            # haverá a metade do número de processos/num_sync_cp_files do que havia na sincronização precedente.
            n_procs = len(list_sync_files)

            if n_procs == 1:
                print('a sincronização total está finalizada. parabéns!')

                # renomeie o arquivo final de checkpoint de sincronização para sync_cp.json
                _sync_filename = list_sync_files[0]
                shutil.move(_sync_filename, 'sync_cp.json')

                make_backup(temp_dir, csv_s_dir)
                return True

            else:
                print(f'iniciando a fusão de conjuntos de símbolos')
                list_sync_cp_dic = get_all_sync_cp_dic(list_sync_files)
                n_procs = n_procs // 2
                pool = mp.Pool(n_procs)
                symbols_to_sync_per_proc = []

                for i in range(n_procs):
                    symbols_to_sync_per_proc.append([])

                for i in range(n_procs * 2):
                    j = math.floor(i / 2)
                    symbols_to_sync_per_proc[j] += list_sync_cp_dic[i]['symbols_to_sync']

                print('removendo sync_cp_files')
                remove_sync_cp_files(get_list_sync_files('.'))

                print('criando novo(s) sync_cp_file(s)')
                for i in range(n_procs):
                    create_sync_cp_file(i, symbols_to_sync_per_proc[i], timeframe)

                dir_sync_l: list[DirectorySynchronization] = []
                for i in range(n_procs):
                    dir_sync = DirectorySynchronization(temp_dir, timeframe, i, symbols_to_sync_per_proc[i])
                    dir_sync_l.append(dir_sync)

                for i in range(n_procs):
                    pool.apply_async(dir_sync_l[i].synchronize_directory, args=(i,))
                pool.close()
                pool.join()

        else:
            print('NEM todos os checkpoints indicam que suas sincronização estão finalizadas')
            print('continuando as sincronizações')
            n_procs = len(list_sync_files)
            pool = mp.Pool(n_procs)

            for i in range(n_procs):
                symbols_to_sync_per_proc.append([])

            for i in range(n_procs):
                symbols_to_sync_per_proc[i] = get_symbols_to_sync(list_sync_files[i])

            dir_sync_l: list[DirectorySynchronization] = []
            for i in range(n_procs):
                dir_sync = DirectorySynchronization(temp_dir, timeframe, i, symbols_to_sync_per_proc[i])
                dir_sync_l.append(dir_sync)

            for i in range(n_procs):
                pool.apply_async(dir_sync_l[i].synchronize_directory, args=(i,))
            pool.close()
            pool.join()

    return False


def find_sync_cache_dir(symbols_to_sync: list[str], root_cache_dir: str):
    reset_dir(root_cache_dir)
    pass


def synchronize_with_cache(symbols_to_sync: list[str] = None) -> bool:
    if not symbols_to_sync:
        return synchronize()

    settings = read_json('settings.json')
    temp_dir = settings['temp_dir']
    csv_o_dir = settings['csv_o_dir']
    csv_s_dir = settings['csv_s_dir']
    root_cache_dir = settings['root_cache_dir']
    timeframe = settings['timeframe']

    reset_dir(temp_dir)

    if symbols_to_sync:
        _len_symbols_to_sync = len(symbols_to_sync)
        if _len_symbols_to_sync == 0:
            print('Não há arquivos CSVs para serem sincronizados.')
            return True
        elif _len_symbols_to_sync == 1:
            print('Há apenas 1 arquivo CSV. Portanto, considera-se que o arquivo já está sincronizado.')
            # make_backup(temp_dir, csv_s_dir)
            cache_dir = find_sync_cache_dir(symbols_to_sync, root_cache_dir)
            copy_files(symbols_to_sync, csv_o_dir, cache_dir)
            copy_files(symbols_to_sync, csv_o_dir, temp_dir)
            return True
        else:
            cache_dir = find_sync_cache_dir(symbols_to_sync, root_cache_dir)
            copy_files(symbols_to_sync, csv_o_dir, cache_dir)
            copy_files(symbols_to_sync, csv_o_dir, temp_dir)

    symbols = search_symbols_in_directory(temp_dir, timeframe)
    _len_symbols = len(symbols)

    if _len_symbols == 0:
        print('Não há arquivos CSVs para serem sincronizados.')
        return True
    elif _len_symbols == 1:
        print('Há apenas 1 arquivo CSV. Portanto, o arquivo será considerado já sincronizado.')
        make_backup(temp_dir, csv_s_dir)
        return True

    symbols_to_sync_per_proc = []
    list_sync_files = get_list_sync_files('.')

    if len(list_sync_files) == 0:
        print('iniciando a sincronização dos arquivos csv pela PRIMEIRA vez.')

        n_procs = choose_n_procs_start(_len_symbols)
        pool = mp.Pool(n_procs)

        for i in range(n_procs):
            symbols_to_sync_per_proc.append([])

        for i in range(len(symbols)):
            i_proc = i % n_procs
            symbols_to_sync_per_proc[i_proc].append(symbols[i])

        dir_sync_l: list[DirectorySynchronization] = []
        for i in range(n_procs):
            dir_sync = DirectorySynchronization(temp_dir, timeframe, i, symbols_to_sync_per_proc[i])
            dir_sync_l.append(dir_sync)

        for i in range(n_procs):
            pool.apply_async(dir_sync_l[i].synchronize_directory, args=(i,))
        pool.close()
        pool.join()

    else:
        print('pode haver sincronização em andamento.')
        print(f'checkpoints: {list_sync_files}')
        # verifique se todos os checkpoints indicam sincronização finalizada
        _results_set = set()
        for i in range(len(list_sync_files)):
            _sync_status = get_sync_status(list_sync_files[i])
            _results_set.add(_sync_status)

        if len(_results_set) == 1 and list(_results_set)[0] is True:
            print('TODOS os checkpoints indicam que suas sincronizações estão FINALIZADAS')
            # assume-se que sempre há um número de processos/num_sync_cp_files que seja uma potência de 2
            # pois será feita uma fusão de cada 2 conjuntos de símbolos de modo que na próxima sincronização
            # haverá a metade do número de processos/num_sync_cp_files do que havia na sincronização precedente.
            n_procs = len(list_sync_files)

            if n_procs == 1:
                print('a sincronização total está finalizada. parabéns!')

                # renomeie o arquivo final de checkpoint de sincronização para sync_cp.json
                _sync_filename = list_sync_files[0]
                shutil.move(_sync_filename, 'sync_cp.json')

                make_backup(temp_dir, csv_s_dir)
                return True

            else:
                print(f'iniciando a fusão de conjuntos de símbolos')
                list_sync_cp_dic = get_all_sync_cp_dic(list_sync_files)
                n_procs = n_procs // 2
                pool = mp.Pool(n_procs)
                symbols_to_sync_per_proc = []

                for i in range(n_procs):
                    symbols_to_sync_per_proc.append([])

                for i in range(n_procs * 2):
                    j = math.floor(i / 2)
                    symbols_to_sync_per_proc[j] += list_sync_cp_dic[i]['symbols_to_sync']

                print('removendo sync_cp_files')
                remove_sync_cp_files(get_list_sync_files('.'))

                print('criando novo(s) sync_cp_file(s)')
                for i in range(n_procs):
                    create_sync_cp_file(i, symbols_to_sync_per_proc[i], timeframe)

                dir_sync_l: list[DirectorySynchronization] = []
                for i in range(n_procs):
                    dir_sync = DirectorySynchronization(temp_dir, timeframe, i, symbols_to_sync_per_proc[i])
                    dir_sync_l.append(dir_sync)

                for i in range(n_procs):
                    pool.apply_async(dir_sync_l[i].synchronize_directory, args=(i,))
                pool.close()
                pool.join()

        else:
            print('NEM todos os checkpoints indicam que suas sincronização estão finalizadas')
            print('continuando as sincronizações')
            n_procs = len(list_sync_files)
            pool = mp.Pool(n_procs)

            for i in range(n_procs):
                symbols_to_sync_per_proc.append([])

            for i in range(n_procs):
                symbols_to_sync_per_proc[i] = get_symbols_to_sync(list_sync_files[i])

            dir_sync_l: list[DirectorySynchronization] = []
            for i in range(n_procs):
                dir_sync = DirectorySynchronization(temp_dir, timeframe, i, symbols_to_sync_per_proc[i])
                dir_sync_l.append(dir_sync)

            for i in range(n_procs):
                pool.apply_async(dir_sync_l[i].synchronize_directory, args=(i,))
            pool.close()
            pool.join()

    return False


def synchronize_loop():
    ret = False
    while not ret:
        ret = synchronize()


def test_01():
    synchronize_loop()


def synchronize_with_cache_loop(symbols: list[str] = None):
    ret = False
    while not ret:
        ret = synchronize_with_cache(symbols)


def test_02():
    synchronize_with_cache_loop()


def test_03():
    currencies_majors_1 = get_symbols('currencies_majors')
    currencies_majors_2 = currencies_majors_1[:]

    currencies_majors_1 = currencies_majors_1[0:-2]

    synchronize_with_cache_loop(currencies_majors_1)
    synchronize_with_cache_loop(currencies_majors_2)


if __name__ == '__main__':
    test_03()
