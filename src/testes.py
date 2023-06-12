import os
import tensorflow as tf


def test_01():
    print(os.environ["LD_LIBRARY_PATH"])
    print(os.environ["PYTHONPATH"])
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))


if __name__ == '__main__':
    test_01()
