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

    p_M5A = Predictors('../predictors/M5A')
    p_M10A = Predictors('../predictors/M10A')
    p_M10B = Predictors('../predictors/M10B')
    p_H1A = Predictors('../predictors/H1A')
    p_H1B = Predictors('../predictors/H1B')

    p_M5A.calculate_outputs(data)
    p_M10A.calculate_outputs(data)
    p_M10B.calculate_outputs(data)
    p_H1A.calculate_outputs(data)
    p_H1B.calculate_outputs(data)

    p_M5A.show_outputs()
    p_M10A.show_outputs()
    p_M10B.show_outputs()
    p_H1A.show_outputs()
    p_H1B.show_outputs()

    p_M5A.show_stats()
    p_M10A.show_stats()
    p_M10B.show_stats()
    p_H1A.show_stats()
    p_H1B.show_stats()

    averages = [p_M5A.average, p_M10A.average, p_M10B.average, p_H1A.average, p_H1B.average]
    total_average = np.average(averages)
    print(f'total_average = {total_average:.2f}')

    return "OK"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
