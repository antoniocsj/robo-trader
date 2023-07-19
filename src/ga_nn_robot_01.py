import os
os.environ['PYTHONHASHSEED'] = str(1)
os.environ['TF_CUDNN_DETERMINISM'] = str(1)
os.environ['TF_DETERMINISTIC_OPS'] = str(1)

import numpy as np
np.random.seed(1)

import random
random.seed(1)

import array
import time

from my_algorithms import eaSimple_WithCP2
from deap import base
from deap import creator
from deap import tools

from utils_train import train_model_param
from utils_filesystem import read_json
from HistMulti import HistMulti


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='B', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_uint", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_uint, 20)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


settings = read_json('settings.json')
print(f'settings.json: {settings}')

csv_dir = settings['csv_dir']
symbol_out = settings['symbol_out']
timeframe = settings['timeframe']
candle_input_type = settings['candle_input_type']
candle_output_type = settings['candle_output_type']

hist = HistMulti(csv_dir, timeframe)
n_features = hist.calc_n_features(candle_input_type)


def individual_to_hyperparameters(ind):
    n_steps = int(''.join(str(x) for x in ind[0:5]), 2)
    if n_steps == 0:
        n_steps = 1

    a = int(''.join(str(x) for x in ind[5:10]), 2)
    b = int(''.join(str(x) for x in ind[10:15]), 2)
    c = int(''.join(str(x) for x in ind[15:20]), 2)

    layer_type = sorted([a, b, c], reverse=True)

    params = {
        'n_steps': n_steps,
        'layer_type': layer_type
    }

    return params


def evaluate(ind):
    params = individual_to_hyperparameters(ind)
    print(params)
    loss = train_model_param(settings, hist, params)
    print(loss)
    time.sleep(10)
    return loss,


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(1)
    freq = 1
    mutpb = 0.2
    n_generations = 30

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10,
    #                                stats=stats, halloffame=hof, verbose=True)
    pop, log, hof = eaSimple_WithCP2(
        population=pop, toolbox=toolbox, checkpoint='checkpoint.pkl', freq=freq,
        cxpb=0.5, mutpb=mutpb, ngen=n_generations,
        stats=stats, halloffame=hof, verbose=True)

    print(f'HallOfFame:')
    print(hof[0])
    print()
    print(individual_to_hyperparameters(hof[0]))
    print()
    print(log)

    return pop, log, hof


if __name__ == "__main__":
    main()
