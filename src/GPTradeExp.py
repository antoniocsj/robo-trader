import operator
import math
import random
import numpy
import itertools

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.tools import HallOfFame

import pygraphviz as pgv
from scoop import futures

# -------------------------------------------------------------------
from TraderSim import TraderSim
from utils import formar_entradas

# configurações para o TraderSim
symbol = 'XAUUSD'
timeframe = 'H1'
initial_deposit = 1000.0
candlesticks_quantity = 50  # quantidade de velas que serão usadas na simulação
close_price_col = 5
trader = TraderSim(symbol, timeframe, initial_deposit)
trader.start_simulation()
trader.previous_price = trader.hist.arr[0, close_price_col]
trader.max_candlestick_count = 5

# -------------------------------------------------------------------
num_velas_anteriores = 5
tipo_vela = 'OHLC'
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
pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(operator.neg, [float], float)
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
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)
pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1), float)
# pset.renameArguments(X='x')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("map", futures.map)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def eval_trade_sim(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    trader.reset()

    index_inicio = num_velas_anteriores
    index_final = index_inicio + candlesticks_quantity

    for i in range(index_inicio, index_final):
        trader.current_price = trader.hist.arr[i, close_price_col]

        print(f'i = {i}, ', end='')
        print(f'OHLCV = {trader.hist.arr[i]}, ', end='')
        print(f'current_price = {trader.current_price:.2f}, ', end='')
        print(f'price_delta = {trader.current_price-trader.previous_price:.2f}')

        trader.update_profit()

        if trader.equity <= 0.0:
            trader.close_position()
            trader.candlestick_count = 0
            trader.finish_simulation()
            print('equity <= 0. a simulação será encerrada.')
            break

        # fecha a posição quando acabarem as novas velas
        if i == candlesticks_quantity - 1:
            trader.close_position()
            trader.candlestick_count = 0
            trader.finish_simulation()
            print('a última vela atingida. a simulação chegou ao fim.')

        if trader.candlestick_count >= trader.max_candlestick_count:
            print(f'fechamento forçado de negociações abertas. a contagem de velas atingiu o limite.')
            trader.close_position()

        if trader.open_position:
            trader.candlestick_count += 1
        else:
            trader.candlestick_count = 0

        trader.print_trade_stats()
        trader.previous_price = trader.current_price

        # ret_msg = trader.interact_with_user()
        entradas = formar_entradas(trader.hist.arr, i, num_velas_anteriores, tipo_vela)
        comando = func(*entradas)

        if comando:
            trader.buy()
        else:
            trader.sell()

    print('\nresultados finais da simulação')
    trader.print_trade_stats()

    return trader.roi,


toolbox.register("evaluate", eval_trade_sim)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    # random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    # mstats.register("avg", numpy.mean)
    # mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    # mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    # print(hof)
    # print(str(gp.PrimitiveTree(hof[0])))
    print(hof[0])
    print_graph(hof)

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


if __name__ == "__main__":
    main()
