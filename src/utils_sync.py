import os
import json


def get_sync_status(filename: str, directory: str = None):
    if directory:
        filepath = f'{directory}/{filename}'
    else:
        filepath = filename

    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            cp = json.load(file)
        finished = cp['finished']
        return finished
    return False


def get_symbols_to_sync(_filename: str) -> list[str] | None:
    if os.path.exists(_filename):
        with open(_filename, 'r') as file:
            cp = json.load(file)
        _symbols = cp['symbols_to_sync']
        return _symbols
    return None


def get_all_sync_cp_dic(_list_sync_files: list[str]) -> list[dict]:
    _list = []
    for _filename in _list_sync_files:
        if os.path.exists(_filename):
            with open(_filename, 'r') as file:
                cp = json.load(file)
                _list.append(cp)
        else:
            print(f'erro em get_all_sync_cp_dic(). arquivo {_filename} não foi encontrado.')
            exit(-1)

    return _list


def create_sync_cp_file(index_proc: int, _symbols_to_sync: list[str], timeframe: str):
    _filename = f'sync_cp_{index_proc}.json'
    _cp = {'id': '',
           'finished': False,
           'current_row': 0,
           'timeframe': timeframe,
           'start': '',
           'end': '',
           'n_symbols': len(_symbols_to_sync),
           'symbols_to_sync': _symbols_to_sync}

    _filename = f'sync_cp_{index_proc}.json'
    with open(_filename, 'w') as file:
        json.dump(_cp, file, indent=4)
    print(f'checkpoint {_filename} criado.')


def remove_sync_cp_files(_list_sync_files: list[str]):
    for filename in _list_sync_files:
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print(f'erro em remove_sync_cp_files(). arquivo {filename} não encontrado.')
            exit(-1)
