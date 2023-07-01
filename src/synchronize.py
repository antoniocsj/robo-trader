import json
from datetime import datetime


def write_json(_filename: str, _dict: dict):
    with open(_filename, 'w') as file:
        json.dump(_dict, file, indent=4)


def read_json(_filename: str) -> dict:
    with open(_filename, 'r') as file:
        _dict = json.load(file)
    return _dict


def synchronize(data: dict):
    """
    Sincroniza as velas.
    :param data:
    :return:
    """
    print('synchronize()')

    last_datetime = datetime.fromisoformat(data['last_datetime'])
    trade_server_datetime = datetime.fromisoformat(data['trade_server_datetime'])
    print(f'last_datetime = {last_datetime}, trade_server_datetime = {trade_server_datetime}')

    timeframe = data['timeframe']
    n_symbols = data['n_symbols']
    rates_count = data['rates_count']
    start_pos = data['start_pos']
    print(f'timeframe = {timeframe}, n_symbols = {n_symbols}, '
          f'rates_count = {rates_count}, start_pos = {start_pos} ')
    
    pass


def test_01():
    data = read_json('request_0.json')
    synchronize(data)


if __name__ == '__main__':
    test_01()
