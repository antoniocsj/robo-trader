from flask import Flask, request
import json

app = Flask(__name__)


def write_json(_filename: str, _dict: dict):
    with open(_filename, 'w') as file:
        json.dump(_dict, file, indent=4)


@app.route('/', methods=['POST'])
def make_prediction():
    print('make_prediction()')
    data = request.json
    print(data)
    write_json('request.json', data)
    return "OK"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
