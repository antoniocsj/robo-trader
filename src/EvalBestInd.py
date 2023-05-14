import operator
import math
import random
import numpy
import itertools
import pickle

from deap import base
from deap import creator
from deap import tools
from deap import gp

from scoop import futures

# -------------------------------------------------------------------
# from TraderSim import TraderSim
from TraderSimNoPrints import TraderSim
from utils import formar_entradas

# configurações para o TraderSim
symbol = 'XAUUSD-SENO'
timeframe = 'M5'
initial_deposit = 1000.0
num_velas_anteriores = 4
tipo_vela = 'OHLC'
max_candlestick_count = 2
trader = TraderSim(symbol, timeframe, initial_deposit)
trader.start_simulation()
close_price_col = 5
trader.previous_price = trader.hist.arr[0, close_price_col]
trader.max_candlestick_count = max_candlestick_count
# candlesticks_quantity é a quantidade de velas que serão usadas na simulação
# candlesticks_quantity = len(trader.hist.arr) - num_velas_anteriores
candlesticks_quantity = 5000
index_inicio = num_velas_anteriores + 50
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
toolbox.register("compile", gp.compile, pset=pset)


def eval_trade_sim_withprints(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    trader.reset()

    for i in range(index_inicio, index_final):
        trader.current_price = trader.hist.arr[i, close_price_col]
        trader.previous_price = trader.hist.arr[i-1, close_price_col]

        print(f'\ni = {i}, ', end='')
        print(f'OHLCV = {trader.hist.arr[i]}, ', end='')
        print(f'current_price = {trader.current_price:.2f}, ', end='')
        print(f'price_delta = {trader.current_price - trader.previous_price:.2f}')

        # fecha a posição quando acabarem as novas velas
        if i == index_final - 1:
            trader.close_position()
            trader.candlestick_count = 0
            trader.finish_simulation()
            print('a última vela foi atingida. a simulação chegou ao fim.')
            break

        entradas = formar_entradas(trader.hist.arr, i, num_velas_anteriores, tipo_vela)
        comando = func(*entradas)
        if comando:
            trader.buy()
        else:
            trader.sell()

        trader.update_profit()
        trader.print_trade_stats()

        if trader.equity <= 0.0:
            trader.close_position()
            trader.candlestick_count = 0
            trader.finish_simulation()
            print('equity <= 0. a simulação será encerrada.')
            break

        if trader.candlestick_count >= trader.max_candlestick_count:
            print(f'fechamento forçado de negociações abertas. a contagem de velas atingiu o limite.')
            trader.close_position()

        if trader.open_position:
            trader.candlestick_count += 1
        else:
            trader.candlestick_count = 0

    print('\nresultados finais da simulação')
    trader.print_trade_stats()

    return trader.roi,


def eval_trade_sim_noprints(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    trader.reset()

    for i in range(index_inicio, index_final):
        trader.current_price = trader.hist.arr[i, close_price_col]
        trader.previous_price = trader.hist.arr[i-1, close_price_col]

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

    return trader.roi,


def read_halloffame():
    with open("halloffame.pkl", "rb") as hof_file:
        cp = pickle.load(hof_file)
        halloffame = cp["halloffame"]
        return halloffame


def main():
    hof = read_halloffame()
    print('obtendo o melhor indivíduo a partir do arquivo halloffame.pkl\n')
    best_ind = hof[0]
    print(best_ind)

    print('\nrodando o TraderSim com o melhor indivíduo.')
    # eval_trade_sim_withprints(best_ind)
    eval_trade_sim_noprints(best_ind)
    print('resultados finais da simulação:\n')
    trader.print_trade_stats()


if __name__ == "__main__":
    main()
