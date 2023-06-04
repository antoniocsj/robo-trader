import operator
import math
import numpy
import itertools
import pickle

import my_algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.tools import HallOfFame

import pygraphviz as pgv
from scoop import futures

# -------------------------------------------------------------------
from TraderSimMultiNoPrints import TraderSimMulti
from utils import formar_entradas

# configurações para a programação genética
n_population = 500
n_generations = 200
max_height = 17
mutpb = 0.1

# configurações para o TraderSim
initial_deposit = 1000.0
trader = TraderSimMulti(initial_deposit)
num_ativos = len(trader.symbols)
num_velas_anteriores = 2
tipo_vela = 'C'
candlesticks_quantity = 500  # quantidade de velas usadas no treinamento

trader.start_simulation()
index_inicio = num_velas_anteriores
index_final = index_inicio + candlesticks_quantity
num_entradas = num_velas_anteriores * len(tipo_vela) * num_ativos

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


# Define a protected log function
def protectedLog(x):
    try:
        return math.log(x)
    except ValueError:
        return 1


# Define a protected exp function
def protectedExp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return 1
    except ValueError:
        return 1


# Define a protected sin function
def protectedSin(x):
    try:
        return math.sin(x)
    except ValueError:
        return 1


# Define a protected cos function
def protectedCos(x):
    try:
        return math.cos(x)
    except ValueError:
        return 1


pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(protectedDiv, [float, float], float, 'div')
pset.addPrimitive(protectedLog, [float], float, 'log')
pset.addPrimitive(protectedExp, [float], float, 'exp')
pset.addPrimitive(max, [float, float], float, 'max')
pset.addPrimitive(min, [float, float], float, 'min')
pset.addPrimitive(protectedCos, [float], float, 'cos')
pset.addPrimitive(protectedSin, [float], float, 'sin')


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
# pset.addEphemeralConstant("pi", lambda: np.pi, float)
# pset.addEphemeralConstant("e", lambda: np.e, float)
# pset.addEphemeralConstant("phi", lambda: (1 + np.sqrt(5))/2, float)
# pset.addEphemeralConstant("rand", lambda: random.random(), float)
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
        print(f'\ni = {i},')
        trader.index = i
        trader.print_symbols_close_price_at(i)
        trader.update_profit()

        entradas = formar_entradas(trader.hist.arr, i, num_velas_anteriores, tipo_vela)
        comando = func(*entradas)
        if comando:
            trader.buy()
        else:
            trader.sell()

        if trader.profit < 0 and abs(trader.profit) / trader.balance >= trader.stop_loss:
            print(f'o stop_loss de {100 * trader.stop_loss:.2f} % for atingido.')
            trader.close_position()

        if trader.equity <= 0.0:
            trader.close_position()
            trader.finish_simulation()
            print('equity <= 0. a simulação será encerrada.')
            break

        # fecha a posição quando acabarem as novas barras (velas ou candlesticks)
        if i == candlesticks_quantity - 1:
            trader.close_position()
            trader.finish_simulation()
            print('a última vela atingida. a simulação chegou ao fim.')

        if trader.candlestick_count >= trader.max_candlestick_count:
            print(f'fechamento forçado de negociações abertas. a contagem de velas atingiu o limite.')
            trader.close_position()

        if trader.open_position[0]:
            trader.candlestick_count += 1
        else:
            trader.candlestick_count = 0

        trader.print_trade_stats()

    print('\nresultados finais da simulação')
    trader.print_trade_stats()

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
        population=pop, toolbox=toolbox, checkpoint='checkpoint.pkl', freq=1,
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
