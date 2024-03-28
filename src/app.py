import tensorflow as tf
from flask import Flask, request
from datetime import datetime
from src.utils.utils_filesystem import write_json
from src.prediction.predictor_utils import load_predictors_groups, execute_predictors_groups


def show_tf():
    print(tf.__version__)
    gpu_list = tf.config.list_physical_devices('GPU')
    if len(gpu_list) == 0:
        print('Não há GPUs instaladas. Abortando.')
        exit(-1)
    else:
        print(gpu_list)


show_tf()
# predictors_groups_paths = ['../predictors_CNN_OHLC', '../predictors_CNN_OHLCV', '../predictors_LSTM']
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
