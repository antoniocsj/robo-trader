from flask import Flask, request
from datetime import datetime
from utils_filesystem import write_json, read_json
from Predictor import Predictor


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

    data = read_json('request_2.json')

    predictor_1 = Predictor(1)
    predictor_1.calc_output(data)

    predictor_2 = Predictor(2)
    predictor_2.calc_output(data)

    predictor_1.show_output()
    predictor_2.show_output()

    return "OK"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
