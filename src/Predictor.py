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
from prediction import prepare_data_for_model


class Predictor:
    def __init__(self):
        self.id = 0
        self.settings = None
        self.train_config = None
        self.scalers = None
        self.model = None

    def load(self, index: int):
        self.id = index

        directory = f'../predictors/{index:02d}'
        if not os.path.exists(directory):
            print(f'ERRO. o diretório {directory} não existe.')
            exit(-1)

        settings_filepath = f'{directory}/settings.json'
        if not os.path.exists(settings_filepath):
            print(f'ERRO. o arquivo {settings_filepath} não existe.')
            exit(-1)
        self.settings = read_json(settings_filepath)

    def output(self, data: dict):
        # procura por predictors no diretório predictors

        # settings = read_json('settings.json')
        symbol_out = self.settings['symbol_out']
        timeframe = self.settings['timeframe']
        _symbol_tf = f'{symbol_out}_{timeframe}'

        # train_config = read_json('train_config.json')

        # with open('scalers.pkl', 'rb') as file:
        #     scalers = pickle.load(file)

        x_input = prepare_data_for_model(data)

        # model = load_model('model.h5')
        output_norm = self.model.predict(x_input)

        bias = self.train_config['bias']
        candle_output_type = self.train_config['candle_output_type']
        scaler = self.scalers[_symbol_tf]

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
