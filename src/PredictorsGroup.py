import os
from utils_filesystem import read_json
from Predictors import Predictors
import numpy as np
from numpy import ndarray


class PredictorsGroup:
    def __init__(self, directory: str):
        self.directory: str = directory
        self.predictors: list[Predictors] = []
        self.averages: ndarray = np.array([])
        self.timeframes: ndarray = np.array([])
        self.stds: ndarray = np.array([])
        self.losses: ndarray = np.array([])

        self.load()

    def load(self):
        all_subdirs = os.listdir(self.directory)

        for subdir in sorted(all_subdirs):
            if subdir.startswith('_') or subdir.endswith('_'):
                continue

            self.predictors.append(Predictors(subdir))

    def calculate_outputs(self, input_data: dict):
        for p in self.predictors:
            p.calculate_outputs(input_data)

        self.averages = np.array([p.output_average for p in self.predictors])
        self.timeframes = np.array([p.timeframe_in_minutes for p in self.predictors])
        self.stds = np.array([p.output_std for p in self.predictors])
        self.losses = np.array([p.losses_average for p in self.predictors])

    def show_outputs(self):
        for p in self.predictors:
            p.show_outputs()

    def show_stats(self):
        for p in self.predictors:
            p.show_stats()

    def show_averages(self):
        inv_timeframes = 1 / self.timeframes
        inv_exp_stds = 1 / np.exp(self.stds)
        inv_exp_losses = 1 / np.exp(self.losses)

        product_1 = inv_timeframes * inv_exp_stds
        product_2 = inv_timeframes * inv_exp_losses
        product_3 = inv_exp_stds * inv_exp_losses
        product_4 = inv_timeframes * inv_exp_stds * inv_exp_losses

        total_avg_1 = np.average(self.averages)
        print(f'total_average_1 (simple) = {total_avg_1:.2f}')

        total_avg_2 = np.average(self.averages, weights=self.timeframes)
        print(f'total_average_2 (weights : timeframes) = {total_avg_2:.2f}')

        total_avg_3 = np.average(self.averages, weights=inv_timeframes)
        print(f'total_average_3 (weights : inv_timeframes) = {total_avg_3:.2f}')

        total_avg_4 = np.average(self.averages, weights=inv_exp_stds)
        print(f'total_average_4 (weights : inv_exp_stds) = {total_avg_4:.2f}')

        total_avg_5 = np.average(self.averages, weights=inv_exp_losses)
        print(f'total_average_5 (weights : inv_exp_losses) = {total_avg_5:.2f}')

        total_avg_6 = np.average(self.averages, weights=product_1)
        print(f'total_average_6 (weights : product_1) = {total_avg_6:.2f}')

        total_avg_7 = np.average(self.averages, weights=product_2)
        print(f'total_average_7 (weights : product_2) = {total_avg_7:.2f}')

        total_avg_8 = np.average(self.averages, weights=product_3)
        print(f'total_average_8 (weights : product_3) = {total_avg_8:.2f}')

        total_avg_9 = np.average(self.averages, weights=product_4)
        print(f'total_average_9 (weights : product_4) = {total_avg_9:.2f}')


if __name__ == '__main__':
    data = read_json('request.json')

    pred_group = PredictorsGroup('../predictors')
    pred_group.calculate_outputs(data)
