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
num_entradas = 25  # 5 velas OHLCV
# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, num_entradas), bool, "IN")

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
pset.renameArguments(IN='x')

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("map", futures.map)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def f(x):
    return x ** 4 + x ** 3 + x ** 2 + x


def eval_trade_sim(individual):
    points = [x / 10. for x in range(-10, 10)]
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    # sqerrors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in points)
    sqerrors = [(func(x) - f(x)) ** 2 for x in points]
    return math.fsum(sqerrors) / len(points),


def evalSpambase(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Randomly sample 400 mails in the spam database
    spam_samp = random.sample(spam, 400)
    # Evaluate the sum of correctly identified mail as spam
    result = sum(bool(func(*mail[:57])) is bool(mail[57]) for mail in spam_samp)
    return result,


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
