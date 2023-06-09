import operator
import math
import random
import numpy as np
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
from GPTradeExp import symbol as symbol_train
from GPTradeExp import timeframe as timeframe_train
from GPTradeExp import candlesticks_quantity as candlesticks_quantity_train
from GPTradeExp import num_velas_anteriores as num_velas_anteriores_train
from GPTradeExp import tipo_vela as tipo_vela_train
from GPTradeExp import toolbox

# configurações para o TraderSim
symbol = symbol_train
timeframe = timeframe_train
initial_deposit = 1000.0
num_velas_anteriores = num_velas_anteriores_train
tipo_vela = tipo_vela_train
candlesticks_quantity = 50000  # quantidade de velas usadas na avaliação

trader = TraderSim(symbol, timeframe, initial_deposit)
trader.start_simulation()
close_price_col = 5
trader.previous_price = trader.hist.arr[0, close_price_col]

index_inicio = num_velas_anteriores + candlesticks_quantity_train
index_final = index_inicio + candlesticks_quantity
num_entradas = num_velas_anteriores * len(tipo_vela)


def eval_trade_sim_withprints(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    trader.reset()

    for i in range(index_inicio, index_final):
        trader.current_price = trader.hist.arr[i, close_price_col]
        trader.previous_price = trader.hist.arr[i - 1, close_price_col]

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
        y = func(*entradas)
        if y >= 1:
            trader.buy()
        elif y <= -1:
            trader.sell()
        elif abs(y) <= 0.5:
            trader.close_position()
        else:
            pass

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

    return trader.hit_rate,


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

    return trader.hit_rate


def read_halloffame():
    with open("halloffame.pkl", "rb") as hof_file:
        cp = pickle.load(hof_file)
        halloffame = cp["halloffame"]
        return halloffame


def main():
    hof = read_halloffame()
    print('avaliação do melhor indivíduo obtido de halloffame.pkl\n')
    best_ind = hof[0]
    print(best_ind)

    print('\nrodando o TraderSim com o melhor indivíduo.')
    # eval_trade_sim_withprints(best_ind)
    eval_trade_sim_noprints(best_ind)
    print('resultados finais da simulação:\n')
    print(f'quantidade de velas usadas na avaliação = {candlesticks_quantity}, '
          f'max_candlestick_count = {max_candlestick_count}')
    trader.print_trade_stats()


if __name__ == "__main__":
    main()
