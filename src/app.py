from flask import Flask, request

app = Flask(__name__)


@app.route('/', methods=['POST'])
def hello_world():
    print('hello_world()')
    data = request.json
    print(data)
    return "OK"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
