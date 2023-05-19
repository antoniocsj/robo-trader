import operator
import math
import random
import numpy
import itertools
import pickle

import numpy as np
from deap import algorithms
import my_algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.tools import HallOfFame

import pygraphviz as pgv
from scoop import futures

# -------------------------------------------------------------------
from TraderSimNoPrints import TraderSim
from utils import formar_entradas

# configurações para a programação genética
n_population = 10000
n_generations = 100
max_height = 40
mutpb = 0.1

# configurações para o TraderSim
symbol = 'XAUUSD-SENO'
timeframe = 'M5'
initial_deposit = 1000.0
num_velas_anteriores = 32
tipo_vela = 'OHLCV'
candlesticks_quantity = 5000  # quantidade de velas usadas no treinamento

trader = TraderSim(symbol, timeframe, initial_deposit)
trader.start_simulation()
close_price_col = 5
trader.previous_price = trader.hist.arr[0, close_price_col]

index_inicio = num_velas_anteriores
index_final = index_inicio + candlesticks_quantity
num_entradas = num_velas_anteriores * len(tipo_vela)

# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, num_entradas), bool, "X")

# Definição de funções que serão usadas na Programação Genética

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)


# floating point operators
# Define a protected division function
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float, 'div')
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(max, [float, float], float, 'max')
pset.addPrimitive(min, [float, float], float, 'min')


pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin, [float], float)


# logic operators
# Define a new if-then-else function
def if_then_else(_input, output1, output2):
    if _input:
        return output1
    else:
        return output2


pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
# pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)
pset.addEphemeralConstant("pi", lambda: np.pi, float)
pset.addEphemeralConstant("rand", lambda: random.random(), float)
# pset.renameArguments(X='x')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
# creator.create("FitnessMaxMin", base.Fitness, weights=(1.0, -1.0))
# creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMaxMin)

toolbox = base.Toolbox()
toolbox.register("map", futures.map)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def eval_trade_sim_noprints(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    trader.reset()

    for i in range(index_inicio, index_final):
        trader.current_price = trader.hist.arr[i, close_price_col]
        trader.previous_price = trader.hist.arr[i - 1, close_price_col]

        # print(f'\ni = {i}, ', end='')
        # print(f'OHLCV = {trader.hist.arr[i]}, ', end='')
        # print(f'current_price = {trader.current_price:.2f}, ', end='')
        # print(f'price_delta = {trader.current_price - trader.previous_price:.2f}')

        # fecha a posição quando acabarem as novas velas
        if i == index_final - 1:
            trader.close_position()
            trader.candlestick_count = 0
            trader.finish_simulation()
            # print('a última vela foi atingida. a simulação chegou ao fim.')
            break

        entradas = formar_entradas(trader.hist.arr, i, num_velas_anteriores, tipo_vela)
        comando = func(*entradas)
        if comando:
            trader.buy()
        else:
            trader.sell()

        trader.update_profit()
        # trader.print_trade_stats()

        if trader.equity <= 0.0:
            trader.close_position()
            trader.candlestick_count = 0
            trader.finish_simulation()
            # print('equity <= 0. a simulação será encerrada.')
            break

        if trader.candlestick_count >= trader.max_candlestick_count:
            # print(f'fechamento forçado de negociações abertas. a contagem de velas atingiu o limite.')
            trader.close_position()

        if trader.open_position:
            trader.candlestick_count += 1
        else:
            trader.candlestick_count = 0

    # print('\nresultados finais da simulação')
    # trader.print_trade_stats()

    return trader.hit_rate,


toolbox.register("evaluate", eval_trade_sim_noprints)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
# toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
# toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))


def main():
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # pop, log = algorithms.eaSimple(population=pop, toolbox=toolbox,
    #                                cxpb=0.5, mutpb=mutpb, ngen=n_generations,
    #                                stats=mstats, halloffame=hof, verbose=True)

    pop, log, hof = my_algorithms.eaSimple_WithCP(
        population=pop, toolbox=toolbox, checkpoint='checkpoint.pkl', freq=2,
        cxpb=0.5, mutpb=mutpb, ngen=n_generations,
        stats=mstats, halloffame=hof, verbose=True)

    print()
    print(log)
    print()
    if len(hof) > 0:
        print(hof[0])
        print_graph(hof)
        print('\nrodando o TraderSim com o melhor indivíduo:')
        eval_trade_sim_noprints(hof[0])
        print('\nresultados finais da simulação')
        trader.print_trade_stats()

        hof_ = read_hof()
        print('\nhof lido de arquivo:')
        print(hof_[0])

    return pop, log, hof


def print_graph(hof: HallOfFame):
    expr = hof[0]
    nodes, edges, labels = gp.graph(expr)

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.pdf")
    write_hof(hof)


def write_hof(hof: HallOfFame):
    cp = dict(halloffame=hof)

    with open("halloffame.pkl", "wb") as hof_file:
        pickle.dump(cp, hof_file)


def read_hof():
    with open("halloffame.pkl", "rb") as hof_file:
        cp = pickle.load(hof_file)
        halloffame = cp["halloffame"]
        return halloffame


if __name__ == "__main__":
    main()
