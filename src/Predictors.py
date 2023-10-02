import os
from utils_filesystem import read_json
from Predictor import Predictor
import numpy as np


class Predictors:
    def __init__(self, directory: str):
        self.directory = directory
        self.predictors: list[Predictor] = []
        self.average = 0.0
        self.std = 0.0  # standard deviation
        self.search_predictors()
        self.all_symbols_trading = False

    def search_predictors(self):
        all_subdirs = os.listdir(self.directory)

        for subdir in sorted(all_subdirs):
            if subdir.startswith('_'):
                continue

            pred = Predictor(subdir, self.directory)
            self.predictors.append(pred)

    def calculate_outputs(self, input_data: dict):
        _sum = 0.0

        outputs = []
        for pred in self.predictors:
            pred.calc_output(input_data, self.all_symbols_trading)
            outputs.append(pred.output)

        self.average = np.average(outputs)
        self.std = np.std(outputs)
        pass

    def show_outputs(self):
        for pred in self.predictors:
            pred.show_output()

    def show_stats(self):
        directory = self.directory.split('/')[2]

        if len(self.predictors) == 0:
            print(f'predictors ({directory}) average: {self.average} std: {self.std}')
        else:
            symbol_out = self.predictors[0].train_config['symbol_out']
            if symbol_out == 'XAUUSD':
                print(f'predictors ({directory}) average: {self.average:.2f} std: {self.std:.2f}')
            else:
                print(f'predictors ({directory}) average: {self.average:.5f} std: {self.std:.2f}')


def teste_01():
    data = read_json('request.json')

    p_01 = Predictors('../predictors/M10A')
    p_02 = Predictors('../predictors/M10B')
    p_09 = Predictors('../predictors/H1A')
    p_10 = Predictors('../predictors/H1B')

    p_01.calculate_outputs(data)
    p_02.calculate_outputs(data)
    p_09.calculate_outputs(data)
    p_10.calculate_outputs(data)

    p_01.show_outputs()
    p_02.show_outputs()
    p_09.show_outputs()
    p_10.show_outputs()

    p_01.show_stats()
    p_02.show_stats()
    p_09.show_stats()
    p_10.show_stats()

    averages = [p_01.average, p_02.average, p_09.average, p_10.average]
    total_average = np.average(averages)
    print(f'total_average = {total_average:.2f}')


if __name__ == '__main__':
    teste_01()
