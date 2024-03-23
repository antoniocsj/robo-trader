import numpy as np
from src.prediction.PredictorsGroup import PredictorsGroup


def load_predictors_groups(paths: list[str]):
    _predictors_groups: list[PredictorsGroup] = []
    for path in paths:
        _predictors_groups.append(PredictorsGroup(path))

    return _predictors_groups


def execute_predictors_groups(_predictors_groups: list[PredictorsGroup], _data: dict):
    outputs = []
    for pred_group in _predictors_groups:
        pred_group.calculate_outputs(_data)

    for pred_group in _predictors_groups:
        pred_group.show_outputs()

    for pred_group in _predictors_groups:
        pred_group.show_stats()

    for pred_group in _predictors_groups:
        pred_group.show_average()
        outputs.append(pred_group.total_average)

    total_average = np.average(outputs)
    print(f'predictors groups total average = {total_average:.2f}')
