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


if __name__ == '__main__':
    test_03()
