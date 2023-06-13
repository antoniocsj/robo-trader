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
    with open("history.json", "r") as file:
        history = json.load(file)
    loss = history['loss']
    val_loss = history['val_loss']
    i_min_loss = np.argmin(loss)
    i_min_val_loss = np.argmin(val_loss)
    print(f'min_loss: {loss[i_min_loss]}, epoch = {i_min_loss}')
    print(f'min_val_loss: {val_loss[i_min_val_loss]}, epoch = {i_min_val_loss}')
    pass


if __name__ == '__main__':
    test_02()
