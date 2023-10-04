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
    p_M5B = Predictors('../predictors/M5B')
    p_M10A = Predictors('../predictors/M10A')
    p_M10B = Predictors('../predictors/M10B')
    p_H1A = Predictors('../predictors/H1A')
    p_H1B = Predictors('../predictors/H1B')

    p_M5A.calculate_outputs(data)
    p_M5B.calculate_outputs(data)
    p_M10A.calculate_outputs(data)
    p_M10B.calculate_outputs(data)
    p_H1A.calculate_outputs(data)
    p_H1B.calculate_outputs(data)

    p_M5A.show_outputs()
    p_M5B.show_outputs()
    p_M10A.show_outputs()
    p_M10B.show_outputs()
    p_H1A.show_outputs()
    p_H1B.show_outputs()

    p_M5A.show_stats()
    p_M5B.show_stats()
    p_M10A.show_stats()
    p_M10B.show_stats()
    p_H1A.show_stats()
    p_H1B.show_stats()

    avg_M5 = np.average([p_M5A.output_average, p_M5B.output_average])
    avg_M10 = np.average([p_M10A.output_average, p_M10B.output_average])
    avg_H1 = np.average([p_H1A.output_average, p_H1B.output_average])

    averages = [avg_M5, avg_M10, avg_H1]
    time_weights = np.array([5, 10, 60])
    # _time_weights = np.array([p_M5A.timeframe_in_minutes, p_M5B.timeframe_in_minutes,
    #                           p_M10A.timeframe_in_minutes, p_M10B.timeframe_in_minutes,
    #                           p_H1A.timeframe_in_minutes, p_H1B.timeframe_in_minutes])
    inv_time_weights = 1 / time_weights

    std_M5 = np.average([p_M5A.output_std, p_M5B.output_std])
    std_M10 = np.average([p_M10A.output_std, p_M10B.output_std])
    std_H1 = np.average([p_H1A.output_std, p_H1B.output_std])
    inv_exp_std_weights = 1 / np.exp([std_M5, std_M10, std_H1])

    inv_time_prod_inv_exp_std_weights = inv_time_weights * inv_exp_std_weights

    total_avg_1 = np.average(averages)
    print(f'total_average_1 (simple) = {total_avg_1:.2f}')

    total_avg_2 = np.average(averages, weights=time_weights)
    print(f'total_average_2 (time_weights) = {total_avg_2:.2f}')

    total_avg_3 = np.average(averages, weights=inv_time_weights)
    print(f'total_average_3 (inv_time_weights) = {total_avg_3:.2f}')

    total_avg_4 = np.average(averages, weights=inv_exp_std_weights)
    print(f'total_average_4 (inv_exp_std_weights) = {total_avg_4:.2f}')

    total_avg_5 = np.average(averages, weights=inv_time_prod_inv_exp_std_weights)
    print(f'total_average_5 (inv_time_prod_inv_exp_std_weights) = {total_avg_5:.2f}')

    total_avg = np.average([total_avg_1, total_avg_2, total_avg_3, total_avg_4, total_avg_5])
    print(f'total_average = {total_avg:.2f}')

    print(f'last_datetime = {last_datetime}, trade_server_datetime = {trade_server_datetime}')

    return "OK"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
