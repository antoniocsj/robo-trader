import os


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

    symbols_names, symbols_paths = search_symbols(csv_dir, timeframe)
    pass


if __name__ == '__main__':
    test_06()
