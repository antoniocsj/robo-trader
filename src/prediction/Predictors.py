import os
from src.utils_filesystem import read_json
from src.prediction.SubPredictor import SubPredictor
import numpy as np


class Predictors:
    def __init__(self, directory: str):
        self.directory = directory
        self.sub_predictors: list[SubPredictor] = []
        self.timeframe = ''
        self.timeframe_in_minutes = 0
        self.output_average = 0.0
        self.output_std = 0.0  # standard deviation
        self.losses_average = 0.0
        self.search_predictors()
        self.all_symbols_trading = False

    def search_predictors(self):
        all_subdirs = os.listdir(self.directory)

        for subdir in sorted(all_subdirs):
            if subdir.startswith('_') or subdir.endswith('_'):
                continue

            pred = SubPredictor(subdir, self.directory)
            print(f'sub_predictor loaded ({pred.id})')

            if self.timeframe and self.timeframe != pred.timeframe:
                print(f'ERRO. timeframes diferentes no mesmo grupo de sub_predictors')
                exit(-1)
            else:
                self.timeframe = pred.timeframe
                self.timeframe_in_minutes = pred.timeframe_in_minutes

            self.sub_predictors.append(pred)

    def calculate_outputs(self, input_data: dict):
        _sum = 0.0

        outputs = []
        losses = []
        for subpred in self.sub_predictors:
            subpred.calc_output(input_data, self.all_symbols_trading)
            outputs.append(subpred.output)
            losses.append(subpred.losses_product)

        self.output_average = np.average(outputs)
        self.output_std = np.std(outputs)
        self.losses_average = np.average(losses)
        pass

    def show_outputs(self):
        for sub_pred in self.sub_predictors:
            sub_pred.show_output()

    def show_stats(self):
        directory = self.directory.split('/')[2]

        if len(self.sub_predictors) == 0:
            print(f'predictors ({directory}) average: {self.output_average} std: {self.output_std}')
        else:
            symbol_out = self.sub_predictors[0].train_config['symbol_out']
            if symbol_out == 'XAUUSD':
                print(f'predictors ({directory}) average: {self.output_average:.2f} std: {self.output_std:.2f}')
            else:
                print(f'predictors ({directory}) average: {self.output_average:.5f} std: {self.output_std:.2f}')


def teste_01():
    data = read_json('request.json')

    p_01 = Predictors('../predictors/H1_OHLC')
    p_02 = Predictors('../predictors/H1_OHLCV')

    p_01.calculate_outputs(data)
    p_02.calculate_outputs(data)

    p_01.show_outputs()
    p_02.show_outputs()

    p_01.show_stats()
    p_02.show_stats()

    averages = [p_01.output_average, p_02.output_average]
    total_average = np.average(averages)
    print(f'total_average = {total_average:.2f}')


if __name__ == '__main__':
    teste_01()
