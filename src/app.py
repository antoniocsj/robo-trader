import numpy as np
import tensorflow as tf
from flask import Flask, request
from datetime import datetime
from utils_filesystem import write_json, read_json
from PredictorsGroup import PredictorsGroup


def show_tf():
    # print(os.environ["LD_LIBRARY_PATH"])
    print(tf.__version__)
    # print(tf.config.list_physical_devices('GPU'))
    print(tf.config.list_physical_devices())


def load_predictors_groups(paths: list[str]):
    _predictors_groups: list[PredictorsGroup] = []
    for path in paths:
        _predictors_groups.append(PredictorsGroup(path))

    return _predictors_groups


def execute_predictors_groups(_predictors_groups: list[PredictorsGroup], _data: dict):
    outputs = []
    for pred_group in _predictors_groups:
        pred_group.calculate_outputs(_data)

    for pred_group in _predictors_groups:
        pred_group.show_outputs()

    for pred_group in _predictors_groups:
        pred_group.show_stats()

    for pred_group in _predictors_groups:
        pred_group.show_average()
        outputs.append(pred_group.total_average)

    total_average = np.average(outputs)
    print(f'predictors groups total average = {total_average:.2f}')


show_tf()
# predictors_groups_paths = ['../predictors_CNN_OHLC', '../predictors']
predictors_groups_paths = ['../predictors']
predictors_groups = load_predictors_groups(predictors_groups_paths)

app = Flask(__name__)


@app.route('/', methods=['POST'])
def make_prediction():
    # print('make_prediction()')
    data = request.json
    # print(data)

    last_datetime = datetime.fromisoformat(data['last_datetime'])
    trade_server_datetime = datetime.fromisoformat(data['trade_server_datetime'])
    print(f'last_datetime = {last_datetime}, trade_server_datetime = {trade_server_datetime}')

    timeframes = data['timeframes']
    symbols = data['symbols']
    n_symbols = data['n_symbols']
    rates_count = data['rates_count']
    start_pos = data['start_pos']

    print(f'timeframes = {timeframes}, n_symbols = {n_symbols}, '
          f'rates_count = {rates_count}, start_pos = {start_pos} ')
    print(f'symbols: {symbols}')

    write_json('request.json', data)

    execute_predictors_groups(predictors_groups, data)

    print(f'last_datetime = {last_datetime}, trade_server_datetime = {trade_server_datetime}')

    return "OK"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
