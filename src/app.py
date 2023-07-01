from flask import Flask, request
import json
from datetime import datetime

app = Flask(__name__)


def write_json(_filename: str, _dict: dict):
    with open(_filename, 'w') as file:
        json.dump(_dict, file, indent=4)


@app.route('/', methods=['POST'])
def make_prediction():
    print('make_prediction()')
    data = request.json
    print(data)

    print(data['last_datetime'])
    _last_datetime = datetime.fromisoformat(data['last_datetime'])
    print(f'last_datetime = {_last_datetime}')

    print(data['trade_server_datetime'])
    _trade_server_datetime = datetime.fromisoformat(data['trade_server_datetime'])
    print(f'trade_server_datetime = {_trade_server_datetime}')

    write_json('request.json', data)
    return "OK"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
