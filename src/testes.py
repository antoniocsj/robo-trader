import numpy as np
from sklearn.preprocessing import MinMaxScaler


def teste_MinMaxScaler():
    data = np.asarray([[10, 0.1],
                       [25, 0.6],
                       [85, 0.47],
                       [36, 0.05],
                       [8, 0.08]])
    print(data)
    scaler: MinMaxScaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    print(data_scaled)
    print()
    new_data = np.asarray([[100, 1],
                           [1, 0.01]])
    print(new_data)
    new_data_scaled = scaler.transform(new_data)
    print(new_data_scaled)


if __name__ == '__main__':
    teste_MinMaxScaler()
