import numpy as np


def test_01():
    import os
    import tensorflow as tf
    print(os.environ["LD_LIBRARY_PATH"])
    print(os.environ["PYTHONPATH"])
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))


def test_02():
    import json
    with open("train_configs.json", "r") as file:
        model_configs = json.load(file)
    loss = model_configs['history']['loss']
    val_loss = model_configs['history']['val_loss']

    i_min_loss = np.argmin(loss)
    min_loss = loss[i_min_loss]
    i_min_val_loss = np.argmin(val_loss)
    min_val_loss = val_loss[i_min_val_loss]

    losses = {'min_loss': {'value': min_loss, 'index': i_min_loss, 'epoch': i_min_loss + 1},
              'min_val_loss': {'value': min_val_loss, 'index': i_min_val_loss, 'epoch': i_min_val_loss + 1}}

    print(f'min_loss: {loss[i_min_loss]}, epoch = {i_min_loss+1}, val_loss = {val_loss[i_min_loss]}')
    print(f'min_val_loss: {val_loss[i_min_val_loss]}, epoch = {i_min_val_loss+1}, loss = {loss[i_min_val_loss]}')
    print(losses)
    pass


if __name__ == '__main__':
    test_02()
