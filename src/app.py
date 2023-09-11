from flask import Flask, request
from datetime import datetime
from utils_filesystem import write_json, read_json
from Predictors import Predictors


app = Flask(__name__)


@app.route('/', methods=['POST'])
def make_prediction():
    # print('make_prediction()')
    data = request.json
    # print(data)

    last_datetime = datetime.fromisoformat(data['last_datetime'])
    trade_server_datetime = datetime.fromisoformat(data['trade_server_datetime'])
    print(f'last_datetime = {last_datetime}, trade_server_datetime = {trade_server_datetime}')

    timeframe = data['timeframe']
    n_symbols = data['n_symbols']
    rates_count = data['rates_count']
    start_pos = data['start_pos']

    print(f'timeframe = {timeframe}, n_symbols = {n_symbols}, '
          f'rates_count = {rates_count}, start_pos = {start_pos} ')

    write_json('request.json', data)
    # predict_next_candle(data)

    # data = read_json('request_2.json')

    p_1 = Predictors('../predictors_2022-08-16_2023-08-18')
    p_1.calculate_outputs(data)

    p_2 = Predictors('../predictors')
    p_2.calculate_outputs(data)

    p_1.show_outputs()
    p_2.show_outputs()

    p_1.show_average()
    p_2.show_average()

    return "OK"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
