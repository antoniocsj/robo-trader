import os
from utils_filesystem import read_json
from Predictor import Predictor


class Predictors:
    def __init__(self, directory: str):
        self.directory = directory
        self.predictors: list[Predictor] = []
        self.average = 0.0
        self.search_predictors()

    def search_predictors(self):
        all_subdirs = os.listdir(self.directory)

        for subdir in sorted(all_subdirs):
            index = int(subdir)
            pred = Predictor(index, self.directory)
            self.predictors.append(pred)

    def calculate_outputs(self, input_data: dict):
        _sum = 0.0

        for pred in self.predictors:
            pred.calc_output(input_data)
            _sum += pred.output

        self.average = _sum / len(self.predictors)

    def show_outputs(self):
        for pred in self.predictors:
            pred.show_output()

    def show_average(self):
        if len(self.predictors) == 0:
            print(f'predictors ({self.directory}) average : {self.average}')
        else:
            symbol_out = self.predictors[0].train_config['symbol_out']
            if symbol_out == 'XAUUSD':
                print(f'predictors ({self.directory}) average : {self.average:.2f}')
            else:
                print(f'predictors ({self.directory}) average : {self.average:.5f}')


def teste_01():
    data = read_json('request.json')

    p = Predictors('../predictors')
    p.calculate_outputs(data)
    p.show_outputs()
    p.show_average()
    pass


if __name__ == '__main__':
    teste_01()
