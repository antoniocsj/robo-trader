
def test_01():
    import os
    import tensorflow as tf

    print(os.environ["LD_LIBRARY_PATH"])
    print(os.environ["PYTHONPATH"])
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))


def test_02():
    import json
    import numpy as np

    with open("train_configs.json", "r") as file:
        train_configs = json.load(file)
    loss = train_configs['history']['loss']
    val_loss = train_configs['history']['val_loss']

    i_min_loss = np.argmin(loss)
    min_loss = loss[i_min_loss]
    i_min_val_loss = np.argmin(val_loss)
    min_val_loss = val_loss[i_min_val_loss]

    losses = {'min_loss': {'value': min_loss, 'index': i_min_loss, 'epoch': i_min_loss + 1},
              'min_val_loss': {'value': min_val_loss, 'index': i_min_val_loss, 'epoch': i_min_val_loss + 1}}

    print(f'min_loss: {loss[i_min_loss]}, epoch = {i_min_loss+1}, val_loss = {val_loss[i_min_loss]}')
    print(f'min_val_loss: {val_loss[i_min_val_loss]}, epoch = {i_min_val_loss+1}, loss = {loss[i_min_val_loss]}')
    print(losses)


def test_03():
    import pickle
    import json
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    class MinMaxScalerEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, MinMaxScaler):
                return obj.__dict__
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(MinMaxScalerEncoder, self).default(obj)

    with open('scalers.pkl', 'rb') as file:
        scalers = pickle.load(file)

    with open('scalers.json', 'w') as file:
        json.dump(scalers, file, indent=4, sort_keys=False, cls=MinMaxScalerEncoder)

    with open('scalers.json', 'r') as file:
        scalers = json.load(file)

    trans: MinMaxScaler = MinMaxScaler()
    trans.__dict__ = scalers['EURUSD_M5']
    print(trans)


def find_max_power2_less_half(n):
    i = 0
    power2 = 2 ** i
    half = n // 2
    while power2 < half:
        i += 1
        power2 = 2 ** i
        if power2 > half:
            i -= 1
            break
    return 2 ** i


def find_max_power2_less_half_with_restriction(n_symbols, max_n_procs):
    p2 = find_max_power2_less_half(n_symbols)
    if p2 > max_n_procs:
        p2 = max_n_procs
    return p2


def test_04():
    # n_cpus = os.cpu_count()
    # n_symbols = 2

    for n_symbols in range(2, 33):
        for n_cpus in range(1, 33):

            max_n_procs = find_max_power2_less_half(n_cpus)
            n_procs = find_max_power2_less_half_with_restriction(n_symbols, max_n_procs)
            print(f'n_symbols = {n_symbols}, n_cpus = {n_cpus}, max_n_procs = {max_n_procs} '
                  f'--> n_procs = {n_procs}')


def test_05():
    import pickle
    import json

    with open('sync_cp_0.pkl', 'rb') as file:
        cp = pickle.load(file)

    _cp = {'finished': cp['finished'],
           'current_row': cp['current_row'],
           'symbols_to_sync': cp['symbols_to_sync']
           }

    with open('sync_cp_0.json', 'w') as file:
        json.dump(_cp, file, indent=4)

    with open('sync_cp_0.json', 'r') as file:
        cp = json.load(file)

    print(cp)


def test_06():
    import itertools as it
    import json
    from utils import search_symbols

    with open('setup.json', 'r') as file:
        setup = json.load(file)
    print(f'setup.json: {setup}')

    csv_dir = setup['csv_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']

    symbols, symbols_paths = search_symbols(csv_dir, timeframe)
    symbols.remove(symbol_out)
    for i in range(2, 5):
        c = list(it.combinations(symbols, i))
    pass


def test_06_1():
    import itertools as it

    min_n_neurons = 0
    max_n_neurons = 270
    step = 30
    _list_n_neurons = list(range(min_n_neurons, max_n_neurons+1, step))
    n_layers = 3
    c = list(it.combinations_with_replacement(_list_n_neurons, n_layers))
    pass


def test_07():
    import json
    import math

    with open('test_models.json', 'r') as file:
        test_models = json.load(file)

    # procurar pelo menor whole_set_train_loss_eval e
    # procurar pelo menor test_loss_eval
    train_loss_min = math.inf
    i_train_loss_min = 0
    test_loss_min = math.inf
    i_test_loss_min = 0
    for i in range(len(test_models)):
        train_loss = test_models[i]['whole_set_train_loss_eval']
        if train_loss < train_loss_min:
            train_loss_min = train_loss
            i_train_loss_min = i
        test_loss = test_models[i]['test_loss_eval']
        if test_loss < test_loss_min:
            test_loss_min = test_loss
            i_test_loss_min = i

    print('train_loss_min:')
    print(test_models[i_train_loss_min])
    print()
    print('test_loss_min:')
    print(test_models[i_test_loss_min])


def test_08():
    import numpy as np
    from numpy import ndarray
    import json
    from HistMulti import HistMulti
    import itertools as it
    from scipy.stats import pearsonr, spearmanr
    from sklearn.preprocessing import MinMaxScaler

    with open('setup.json', 'r') as file:
        setup = json.load(file)
    print(f'setup.json: {setup}')

    csv_dir = setup['csv_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']
    hist = HistMulti(csv_dir, timeframe)

    pairs = list(it.combinations(hist.symbols, 2))
    _set = set()
    corr_min_abs = 0.9
    for symbol_a, symbol_b in pairs:
        data1: ndarray = hist.arr[f'{symbol_a}_{timeframe}'][:, 4]
        _data1 = data1.tolist()
        data2: ndarray = hist.arr[f'{symbol_b}_{timeframe}'][:, 4]
        _data2 = data2.tolist()
        pearsons_corr, _ = pearsonr(_data1, _data2)
        spearmans_corr, _ = spearmanr(_data1, _data2)
        if abs(pearsons_corr) > corr_min_abs and abs(spearmans_corr) > corr_min_abs:
            print(f'{symbol_a}-{symbol_b}: correlations: Pearsons = {pearsons_corr:.3f}, '
                  f'Spearmans = {spearmans_corr:.3f}')
            if 'EURUSD' in symbol_a or 'EURUSD' in symbol_b:
                _set.add(symbol_a)
                _set.add(symbol_b)

    print(len(_set))
    print(_set)


def test_09():
    import numpy as np

    yr = np.array([1, 2, 3, 4, 5])
    yp = np.array([1.2, 2.3, 3.1, 4.25, 5.35])
    diffs = []

    for i in range(len(yr)):
        diff = yr[i] - yp[i]
        diffs.append(diff)

    diffs = np.asarray(diffs)
    bias = np.sum(diffs) / len(diffs)
    print(f'bias = {bias}')


if __name__ == '__main__':
    test_09()
