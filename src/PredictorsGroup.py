import os
from utils_filesystem import read_json
from Predictors import Predictors
import numpy as np
from numpy import ndarray


class PredictorsGroup:
    def __init__(self, directory: str):
        self.directory: str = directory
        self.predictors: list[Predictors] = []
        self.outputs: ndarray = np.array([])
        self.timeframes: ndarray = np.array([])
        self.stds: ndarray = np.array([])
        self.losses: ndarray = np.array([])

        self.load()

    def load(self):
        all_subdirs = os.listdir(self.directory)

        for subdir in sorted(all_subdirs):
            if subdir.startswith('_') or subdir.endswith('_'):
                continue

            sub_pred_dir = f'{self.directory}/{subdir}'
            self.predictors.append(Predictors(sub_pred_dir))

    def calculate_outputs(self, input_data: dict):
        for p in self.predictors:
            p.calculate_outputs(input_data)

        self.outputs = np.array([p.output_average for p in self.predictors])
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
        inv_losses = 1 / self.losses

        product_1 = inv_timeframes * inv_exp_stds
        product_2 = inv_timeframes * inv_losses
        product_3 = inv_exp_stds * inv_losses
        product_4 = inv_timeframes * inv_exp_stds * inv_losses
        product_5 = self.timeframes * inv_losses

        total_avg_1 = np.average(self.outputs, weights=inv_timeframes)
        print(f'total_average_1 (weights : inv_timeframes) = {total_avg_1:.2f}')

        total_avg_2 = np.average(self.outputs, weights=inv_exp_stds)
        print(f'total_average_2 (weights : inv_exp_stds) = {total_avg_2:.2f}')

        total_avg_3 = np.average(self.outputs, weights=inv_losses)
        print(f'total_average_3 (weights : inv_losses) = {total_avg_3:.2f}')

        total_avg_4 = np.average(self.outputs, weights=product_1)
        print(f'total_average_4 (weights : inv_timeframes * inv_exp_stds) = {total_avg_4:.2f}')

        total_avg_5 = np.average(self.outputs, weights=product_2)
        print(f'total_average_5 (weights : inv_timeframes * inv_losses) = {total_avg_5:.2f}')

        total_avg_6 = np.average(self.outputs, weights=product_3)
        print(f'total_average_6 (weights : inv_exp_stds * inv_losses) = {total_avg_6:.2f}')

        total_avg_7 = np.average(self.outputs, weights=product_4)
        print(f'total_average_7 (weights : inv_timeframes * inv_exp_stds * inv_losses) = {total_avg_7:.2f}')

        total_avg_8 = np.average(self.outputs, weights=product_5)
        print(f'total_average_8 (weights : timeframes * inv_losses) = {total_avg_8:.2f}')

        averages = [total_avg_1, total_avg_2, total_avg_3, total_avg_4, total_avg_5, total_avg_6, total_avg_7,
                    total_avg_8]

        total_avg = np.average(averages)
        total_avg_std = np.std(averages)
        print(f'total_average = {total_avg:.2f} std = {total_avg_std:.2f}')


if __name__ == '__main__':
    data = read_json('request.json')

    pred_group = PredictorsGroup('../predictors')
    pred_group.calculate_outputs(data)
    pred_group.show_outputs()
    pred_group.show_stats()
    pred_group.show_averages()
    pass
