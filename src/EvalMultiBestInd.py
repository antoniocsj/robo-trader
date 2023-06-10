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
from TraderSimMultiNoPrints import TraderSimMulti
from utils import formar_entradas_multi
from GPMultiTradeExp import candlesticks_quantity as candlesticks_quantity_train
from GPMultiTradeExp import num_velas_anteriores as num_velas_anteriores_train
from GPMultiTradeExp import tipo_vela as tipo_vela_train
from GPMultiTradeExp import num_ativos as num_ativos_train
from GPMultiTradeExp import toolbox

# configurações para o TraderSim
initial_deposit = 1000.0
trader = TraderSimMulti(initial_deposit)
num_ativos = num_ativos_train
num_velas_anteriores = num_velas_anteriores_train
tipo_vela = tipo_vela_train
candlesticks_quantity = 50000  # quantidade de velas usadas na avaliação

trader.start_simulation()
index_inicio = num_velas_anteriores + candlesticks_quantity_train
index_final = index_inicio + candlesticks_quantity
num_entradas = num_velas_anteriores * len(tipo_vela) * num_ativos


def eval_trade_sim_noprints(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    trader.reset()

    for i in range(index_inicio, index_final):
        # print(f'\ni = {i},')
        trader.index = i
        # trader.print_symbols_close_price_at(i)
        trader.update_profit()

        entradas = formar_entradas_multi(trader.hist, i, num_velas_anteriores, tipo_vela)
        y = func(*entradas)
        _y = int(np.clip(np.round(np.abs(y)), 0, num_ativos - 1))
        _symbol = trader.symbols[_y]

        if y >= 1:
            trader.buy(_symbol)
        elif y <= -1:
            trader.sell(_symbol)
        elif np.abs(y) <= 0.5:
            trader.close_position()
        else:
            pass

        if trader.profit < 0 and abs(trader.profit) / trader.balance >= trader.stop_loss:
            # print(f'o stop_loss de {100 * trader.stop_loss:.2f} % for atingido.')
            trader.close_position()

        if trader.equity <= 0.0:
            trader.close_position()
            trader.finish_simulation()
            # print('equity <= 0. a simulação será encerrada.')
            break

        # fecha a posição quando acabarem as novas barras (velas ou candlesticks)
        if i == candlesticks_quantity - 1:
            trader.close_position()
            trader.finish_simulation()
            # print('a última vela atingida. a simulação chegou ao fim.')

        if trader.candlestick_count >= trader.max_candlestick_count:
            # print(f'fechamento forçado de negociações abertas. a contagem de velas atingiu o limite.')
            trader.close_position()

        if trader.open_position[0]:
            trader.candlestick_count += 1
        else:
            trader.candlestick_count = 0

        # trader.print_trade_stats()

    # print('\nresultados finais da simulação')
    # trader.print_trade_stats()

    return trader.hit_rate,


def eval_trade_sim_withprints(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    trader.reset()

    for i in range(index_inicio, index_final):
        print(f'\ni = {i},')
        trader.index = i
        trader.print_symbols_close_price_at(i)
        trader.update_profit()

        entradas = formar_entradas_multi(trader.hist, i, num_velas_anteriores, tipo_vela)
        y = func(*entradas)
        _y = int(np.clip(np.round(np.abs(y)), 0, num_ativos - 1))
        _symbol = trader.symbols[_y]

        if y >= 1:
            trader.buy(_symbol)
        elif y <= -1:
            trader.sell(_symbol)
        elif np.abs(y) <= 0.5:
            trader.close_position()
        else:
            pass

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
    print(f'quantidade de velas usadas na avaliação = {candlesticks_quantity}')
    trader.print_trade_stats()


if __name__ == "__main__":
    main()
