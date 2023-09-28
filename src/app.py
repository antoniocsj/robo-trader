import numpy as np
import tensorflow as tf
from flask import Flask, request
from datetime import datetime
from utils_filesystem import write_json, read_json
from Predictors import Predictors


def show_tf():
    # print(os.environ["LD_LIBRARY_PATH"])
    print(tf.__version__)
    # print(tf.config.list_physical_devices('GPU'))
    print(tf.config.list_physical_devices())


show_tf()
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
    # predict_next_candle(data)

    # data = read_json('request.json')

    p_01 = Predictors('../predictors/M10A')
    p_02 = Predictors('../predictors/M10B')
    p_09 = Predictors('../predictors/H1A')
    p_10 = Predictors('../predictors/H1B')

    p_01.calculate_outputs(data)
    p_02.calculate_outputs(data)
    p_09.calculate_outputs(data)
    p_10.calculate_outputs(data)

    p_01.show_outputs()
    p_02.show_outputs()
    p_09.show_outputs()
    p_10.show_outputs()

    p_01.show_stats()
    p_02.show_stats()
    p_09.show_stats()
    p_10.show_stats()

    averages = [p_01.average, p_02.average, p_09.average, p_10.average]
    total_average = np.average(averages)
    print(f'total_average = {total_average:.2f}')

    return "OK"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
