import os
os.environ['PYTHONHASHSEED'] = str(1)
os.environ['TF_CUDNN_DETERMINISM'] = str(1)
os.environ['TF_DETERMINISTIC_OPS'] = str(1)

import numpy as np
np.random.seed(1)

import random
random.seed(1)

import array as ar
import time

from my_algorithms import eaSimple_WithCP2
from deap import base
from deap import creator
from deap import tools

from utils_train import train_model_param
from utils_filesystem import read_json
from utils_symbols import get_symbols
from HistMulti import HistMulti


individual_size = 87


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", ar.array, typecode='B', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_uint", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_uint, individual_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


settings = read_json('settings.json')
print(f'settings.json: {settings}')

temp_dir = settings['temp_dir']
timeframe = settings['timeframe']
candle_input_type = settings['candle_input_type']
candle_output_type = settings['candle_output_type']

hist = HistMulti(temp_dir, timeframe)
n_features = hist.calc_n_features(candle_input_type)


def make_layers_config(max_n_layers: int,
                       n_bits_per_layer: int,
                       layers_bit_start: int,
                       ind: ar.array) -> list[int]:
    a = layers_bit_start
    b = a + n_bits_per_layer

    if layers_bit_start + max_n_layers * n_bits_per_layer > len(ind):
        print('ERRO. layers_bit_start + n_layers * n_bits_per_layer > len(ind)')
        exit(-1)

    _list = []
    for i in range(max_n_layers):
        x = int(''.join(str(x) for x in ind[a:b]), 2)
        _list.append(x)
        a, b = a + n_bits_per_layer, b + n_bits_per_layer

    _list = sorted(_list, reverse=True)
    return _list


def get_symbols_from_bits_segment(segment: ar.array, all_symbols: list[str]) -> list[str]:
    if len(segment) != len(all_symbols):
        print('ERRO. get_symbols_from_bits_segment(). len(segment) != len(all_symbols)')
        exit(-1)

    _list = []
    for x, y in zip(segment, all_symbols):
        if x:
            _list.append(y)
    return _list


def get_symbolout_from_bits_segment(segment: ar.array, all_symbols: list[str]) -> str:
    if 2 ** len(segment) < len(all_symbols):
        print('ERRO. (len(segment) ** 2) - 1 < len(all_symbols)')
        exit(-1)

    symbol_idx = int(''.join(str(x) for x in segment), 2)

    if symbol_idx > len(all_symbols) - 1:
        symbol_idx = len(all_symbols) - 1

    symbol = all_symbols[symbol_idx]

    return symbol


def individual_to_hyperparameters(ind):
    # a escolha dos símbolos será assim:
    # haverão 45 bits (ex. 001001100101001....100101) cada bit representa um símbolo de uma lista
    # ordenada alfabeticamente de símbolos. quando o bit é 0, o símbolos não estão presente, e quando o bit
    # é 1, o símbolo está presente.
    # Também haverão alguns outros bits que representam um número de 1 a 45, que representará o symbol_out.

    n_steps_bit_start = 0
    n_steps_bits_len = 4
    n_steps_bit_end = n_steps_bit_start + n_steps_bits_len
    n_steps = int(''.join(str(x) for x in ind[n_steps_bit_start:n_steps_bit_end]), 2)
    n_steps += 1

    max_n_layers = 4
    n_bits_per_layer = 8
    layers_bit_start = n_steps_bit_end
    layers_bit_end = layers_bit_start + max_n_layers * n_bits_per_layer

    layers_config = make_layers_config(max_n_layers, n_bits_per_layer, layers_bit_start, ind)

    all_symbols = get_symbols()

    symbol_out_bit_start = layers_bit_end
    symbol_out_bits_len = 6
    symbol_out_bit_end = symbol_out_bit_start + symbol_out_bits_len
    symbol_out = get_symbolout_from_bits_segment(ind[symbol_out_bit_start:symbol_out_bit_end], all_symbols)

    symbols_bits_start = symbol_out_bit_end
    symbols_bits_len = len(all_symbols)  # 45
    symbols_bits_end = symbols_bits_start + symbols_bits_len
    symbols = get_symbols_from_bits_segment(ind[symbols_bits_start:symbols_bits_end], all_symbols)

    if len(symbols) == 0:
        symbols = [symbol_out]

    params = {
        'n_steps': n_steps,
        'layers_config': layers_config,
        'symbol_out': symbol_out,
        'symbols': symbols
    }

    return params


def evaluate(ind):
    params = individual_to_hyperparameters(ind)
    print(params)
    # loss = train_model_param(settings, hist, params)
    # print(loss)
    # time.sleep(30)
    loss = sum(ind)
    return loss,


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(1)
    freq = 1
    mutpb = 0.2
    n_generations = 40

    pop = toolbox.population(n=100)
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
