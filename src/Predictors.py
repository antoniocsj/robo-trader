import os
from utils_filesystem import read_json
from Predictor import Predictor


class Predictors:
    def __int__(self, directory: str):
        print('###')
        self.directory = directory
        self.predictors: list[Predictor] = []
        self.search_predictors()

    def search_predictors(self):
        all_subdirs = os.listdir(self.directory)

        for subdir in all_subdirs:
            index = int(subdir)
            pred = Predictor(index)
            self.predictors.append(pred)

    def calculate_outputs(self, input_data: dict):
        for pred in self.predictors:
            pred.calc_output(input_data)

    def show_outputs(self):
        for pred in self.predictors:
            pred.show_output()


def teste_01():
    data = read_json('request_2.json')

    p = Predictors()
    p.calculate_outputs(data)
    p.show_outputs()
    pass


if __name__ == '__main__':
    teste_01()
