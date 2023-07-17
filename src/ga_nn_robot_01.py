import os
os.environ['PYTHONHASHSEED'] = str(1)
os.environ['TF_CUDNN_DETERMINISM'] = str(1)
os.environ['TF_DETERMINISTIC_OPS'] = str(1)

import numpy as np
np.random.seed(1)

import random
random.seed(1)

import array

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='B', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_uint", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_uint, 20)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(ind):
    a = int(''.join(str(x) for x in ind[0:5]), 2)
    b = int(''.join(str(x) for x in ind[5:10]), 2)
    c = int(''.join(str(x) for x in ind[10:15]), 2)
    d = int(''.join(str(x) for x in ind[15:20]), 2)
    pass
    return a*b/(c+1)-c//(a+d+1),


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(1)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
                                   stats=stats, halloffame=hof, verbose=True)
    print(f'HallOfFame:')
    print(hof[0])

    return pop, log, hof


if __name__ == "__main__":
    main()
