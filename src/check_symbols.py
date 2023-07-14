from utils_filesystem import read_json
from utils_symbols import search_symbols
from Sheet import Sheet
from HistMulti import HistMulti


def check_symbols_timedelta():
    settings = read_json('settings.json')
    print(f'settings.json: {settings}')

    csv_dir = settings['csv_dir']
    timeframe = settings['timeframe']

    hist = HistMulti(csv_dir, timeframe)

    for symbol in hist.symbols:
        print(symbol)


if __name__ == '__main__':
    check_symbols_timedelta()
