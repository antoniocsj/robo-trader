from datetime import datetime
from src.utils.utils_filesystem import read_json
from Sheet import Sheet
from HistMulti import HistMulti


def check_symbols_timedelta():
    """
    Verifica se todas as linhas estão de cada símbolo (arquivo CSV) com a data/hora em ordem crescente.
    Se houver alguma linha que desobedece essa regra, reporta o evento.
    :return:
    """
    settings = read_json('settings.json')
    print(f'settings.json: {settings}')

    temp_dir = settings['temp_dir']
    timeframe = settings['timeframe']

    hist = HistMulti(temp_dir, timeframe)

    s: Sheet
    for s in list(hist.sheets.values()):
        print(s.symbol)
        row = s.df.iloc[0]
        datetime_str = row['DATETIME']
        datetime_row_0 = datetime.fromisoformat(datetime_str)

        row = s.df.iloc[1]
        datetime_str = row['DATETIME']
        datetime_row_1 = datetime.fromisoformat(datetime_str)

        if datetime_row_1 != datetime_row_0 + s.timedelta:
            print(f'ERRO em {s.symbol}. o timedelta ({s.timedelta}) não é respeita entre as linha 0 e 1.')
            print(f'datetime linha 0 = {datetime_row_0}, datetime da linha 1 = {datetime_row_1}')
            exit(-1)


if __name__ == '__main__':
    check_symbols_timedelta()
