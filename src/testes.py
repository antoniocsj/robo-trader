import math


def test_01():
    import os
    import tensorflow as tf

    print(os.environ["LD_LIBRARY_PATH"])
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

    print(f'min_loss: {loss[i_min_loss]}, epoch = {i_min_loss + 1}, val_loss = {val_loss[i_min_loss]}')
    print(f'min_val_loss: {val_loss[i_min_val_loss]}, epoch = {i_min_val_loss + 1}, loss = {loss[i_min_val_loss]}')
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
    from utils_symbols import search_symbols

    with open('settings.json', 'r') as file:
        setup = json.load(file)
    print(f'settings.json: {setup}')

    temp_dir = setup['temp_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']

    symbols, symbols_paths = search_symbols(temp_dir, timeframe)
    symbols.remove(symbol_out)
    for i in range(2, 5):
        c = list(it.combinations(symbols, i))
    pass


def test_06_1():
    import itertools as it

    min_n_neurons = 0
    max_n_neurons = 270
    step = 30
    _list_n_neurons = list(range(min_n_neurons, max_n_neurons + 1, step))
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

    with open('settings.json', 'r') as file:
        setup = json.load(file)
    print(f'settings.json: {setup}')

    temp_dir = setup['temp_dir']
    symbol_out = setup['symbol_out']
    timeframe = setup['timeframe']
    hist = HistMulti(temp_dir, timeframe)

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


def test_10():
    import array
    a = array.array('B', [0, 1, 0, 1, 1])
    b = ['a', 'b', 'c', 'd', 'e']
    _list = []
    for x, y in zip(a, b):
        if x:
            _list.append(y)
    print(_list)


def test_11():
    import itertools as it
    from utils_symbols import get_symbols
    symbols = get_symbols('currencies_majors')
    combs = []
    for i in range(1, len(symbols) + 1):
        comb = list(it.combinations(symbols, i))
        combs.append(comb)
    pass


def load_train_log() -> dict:
    import os
    from utils_filesystem import read_json
    _filename = 'train_log.json'
    if os.path.exists(_filename):
        _dict = read_json(_filename)
        return _dict
    else:
        print(f'ERRO. o arquivo {_filename} não existe.')
        exit(-1)


def test_12():
    train_log = load_train_log()
    experiments = train_log['experiments']

    min_test_loss = math.inf
    min_whole_set_train_loss = math.inf
    min_product = math.inf
    random_seed_min_test_loss = -1
    random_seed_min_whole_set_train_loss = -1
    random_seed_min_product = -1

    for e in experiments:
        test_loss_eval = e['test_loss_eval']
        whole_set_train_loss_eval = e['whole_set_train_loss_eval']
        product = e['product']

        if whole_set_train_loss_eval < min_whole_set_train_loss:
            min_whole_set_train_loss = whole_set_train_loss_eval
            random_seed_min_whole_set_train_loss = e['random_seed']

        if test_loss_eval < min_test_loss:
            min_test_loss = test_loss_eval
            random_seed_min_test_loss = e['random_seed']

        if product < min_product:
            min_product = product
            random_seed_min_product = e['random_seed']

    print(f'random_seed_min_test_loss = {random_seed_min_test_loss}')
    elem = experiments[random_seed_min_test_loss - 1]
    print(f'{elem}\n')
    print(f'random_seed_min_whole_set_train_loss = {random_seed_min_whole_set_train_loss}')
    elem = experiments[random_seed_min_whole_set_train_loss - 1]
    print(f'{elem}\n')
    print(f'random_seed_min_product = {random_seed_min_product}')
    elem = experiments[random_seed_min_product - 1]
    print(f'{elem}\n')


def print_list(_list: list):
    for i in range(len(_list)):
        print(f'{i:03d} {_list[i]}')
    print('')


def test_13():
    from operator import itemgetter, attrgetter

    train_log = load_train_log()
    experiments = train_log['experiments']
    sorted_exps_by_test_loss = sorted(experiments, key=lambda d: d['test_loss'])
    sorted_exps_by_whole_set_train_loss = sorted(experiments, key=lambda d: d['whole_set_train_loss'])
    sorted_exps_by_product_losses = sorted(experiments, key=lambda d: d['product'])

    print('sorted_exps_by_product_losses:')
    print_list(sorted_exps_by_product_losses)

    print('sorted_exps_by_test_loss:')
    print_list(sorted_exps_by_test_loss)

    print('sorted_exps_by_whole_set_train_loss:')
    print_list(sorted_exps_by_whole_set_train_loss)


def test_14():
    from sklearn.datasets import make_blobs
    from matplotlib import pyplot
    from pandas import DataFrame

    # generate 2d classification dataset
    n_samples = 33058
    n_features = 60
    n_clusters = 3
    X, y = make_blobs(n_samples, n_features, centers=n_clusters, random_state=1)

    # scatter plot, dots colored by class value
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    fig, ax = pyplot.subplots()

    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

    pyplot.show()


if __name__ == '__main__':
    test_14()
