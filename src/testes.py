import numpy as np
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
index_inicio = num_velas_anteriores
index_final = index_inicio + candlesticks_quantity
for i in range(index_inicio, index_final):
    print(f'i = {i}')
    entradas = formar_entradas(trader.hist.arr, i, num_velas_anteriores, tipo_vela)
    print(*entradas)
