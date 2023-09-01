import json
import os.path
import pickle
import numpy as np
from numpy import ndarray
from datetime import datetime
from Sheet import Sheet
from HistMulti import HistMulti
from utils_filesystem import read_json
from utils_symbols import search_symbols_in_dict
from utils_nn import prepare_data_for_prediction
from utils_ops import denorm_output
from setups import apply_setup_symbols
from keras.models import load_model


class Predictor:
    def __int__(self):
        self.settings = None
        self.train_config = None
        self.scalers = None
        self.model = None

    def load(self, index: int):
        directory = f'../predictors/{index:02d}'
        if os.path.exists(directory):
            pass


def predictor_output(data: dict, settings, train_config, scalers, model):
    # procura por predictors no diretório predictors

    # settings = read_json('settings.json')
    symbol_out = settings['symbol_out']
    timeframe = settings['timeframe']
    _symbol_tf = f'{symbol_out}_{timeframe}'

    # train_config = read_json('train_config.json')

    # with open('scalers.pkl', 'rb') as file:
    #     scalers = pickle.load(file)

    x_input = prepare_data_for_model(data)

    # model = load_model('model.h5')
    output_norm = model.predict(x_input)

    bias = train_config['bias']
    candle_output_type = train_config['candle_output_type']
    scaler = scalers[_symbol_tf]

    print('considerando o bias(+):')
    output_denorm = denorm_output(output_norm, bias, candle_output_type, scaler)
    # print(f'previsão para a próxima vela: {candle_output_type} = {output_denorm}')
    if len(candle_output_type) == 1:
        print(f'previsão para a próxima vela: {candle_output_type} = {output_denorm:.5f}')
    else:
        print(f'previsão para a próxima vela: {candle_output_type} = {output_denorm}')

    if candle_output_type == 'OHLC' or candle_output_type == 'OHLCV':
        dCO = output_denorm[3] - output_denorm[0]
        print(f'C - O = {dCO:.5f}')

    print('considerando o bias=0:')
    output_denorm = denorm_output(output_norm, 0.0, candle_output_type, scaler)
    # print(f'previsão para a próxima vela: {candle_output_type} = {output_denorm}')
    if len(candle_output_type) == 1:
        print(f'previsão para a próxima vela: {candle_output_type} = {output_denorm:.5f}')
    else:
        print(f'previsão para a próxima vela: {candle_output_type} = {output_denorm}')

    if candle_output_type == 'OHLC' or candle_output_type == 'OHLCV':
        dCO = output_denorm[3] - output_denorm[0]
        print(f'C - O = {dCO:.5f}')

    print('considerando o bias(-):')
    bias = (-np.array(bias)).tolist()
    output_denorm = denorm_output(output_norm, bias, candle_output_type, scaler)
    # print(f'previsão para a próxima vela: {candle_output_type} = {output_denorm}')
    if len(candle_output_type) == 1:
        print(f'previsão para a próxima vela: {candle_output_type} = {output_denorm:.5f}')
    else:
        print(f'previsão para a próxima vela: {candle_output_type} = {output_denorm}')

    if candle_output_type == 'OHLC' or candle_output_type == 'OHLCV':
        dCO = output_denorm[3] - output_denorm[0]
        print(f'C - O = {dCO:.5f}')


def teste_01():
    pass


if __name__ == '__main__':
    teste_01()
