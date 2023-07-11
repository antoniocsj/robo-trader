import os
import pickle
from HistMulti import HistMulti
from sklearn.preprocessing import MinMaxScaler
from utils_filesystem import read_json
from utils_ops import denorm_close_price


class TraderSimMulti:
    def __init__(self, initial_deposit: float):
        setup = read_json('settings.json')
        print(f'settings.json: {setup}')

        csv_dir = setup['csv_dir']
        timeframe = setup['timeframe']

        self.symbols = []  # financial assets
        self.timeframe = timeframe
        self.hist = HistMulti(csv_dir, timeframe)
        self.open_position = ('', '')
        self.candlestick_count = 0  # contagem de velas desde a abertura da posição
        self.max_candlestick_count = 5  # contagem máxima permitida de velas desde a abertura da posição
        self.simulation_is_running = False
        self.index = 0
        self.num_hits = 0  # número de acertos
        self.num_misses = 0  # número de erros
        self.num_buys = 0  # número de negociações de compra
        self.num_sells = 0  # número de negociações de venda
        self.num_trades = 0  # número de negociações de compra ou venda
        self.hit_rate = 0.0  # taxa de acertos
        self.current_price = 0.0  # preço atual ou o preço de fechamento atual
        self.previous_price = 0.0  # preço anterior ou o preço de fechamento da vela anterior
        self.starting_price = 0.0
        self.final_price = 0.0
        self.initial_balance = initial_deposit  # saldo inicial
        self.balance = self.initial_balance  # saldo atual
        self.equity = self.initial_balance  # patrimônio líquido
        self.profit = 0.0  # lucro (ou prejuízo) na negociação
        self.contract_size = 100000  # Forex Majors=100000, XAUUSD=100
        self.volume_operation = 0.01
        self.roi = 0.0  # Return on Investment ou Retorno de Investmento
        self.stop_loss = 0.01  # limiar de percentual de perda máxima por negociação
        self.scalers = None
        self.load_symbols()
        self.load_scalers()

    def load_symbols(self):
        self.symbols = self.hist.symbols[:]
        self.timeframe = self.hist.timeframe

    def load_scalers(self):
        if os.path.exists('scalers.pkl'):
            with open('scalers.pkl', 'rb') as file:
                self.scalers = pickle.load(file)

    def reset(self):
        self.open_position = ('', '')
        self.candlestick_count = 0
        self.simulation_is_running = False
        self.index = 0
        self.num_hits = 0
        self.num_misses = 0
        self.num_buys = 0
        self.num_sells = 0
        self.num_trades = 0
        self.hit_rate = 0.0
        self.current_price = 0.0
        self.previous_price = 0.0
        self.starting_price = 0.0
        self.final_price = 0.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.profit = 0.0
        self.roi = 0.0
        self.start_simulation()

    @property
    def num_trades(self):
        return self.num_hits + self.num_misses

    @num_trades.setter
    def num_trades(self, value):
        self._num_trades = value

    @property
    def hit_rate(self):
        if self.num_trades == 0:
            return 0.0
        else:
            return self.num_hits / self.num_trades

    @hit_rate.setter
    def hit_rate(self, value):
        self._hit_rate = value

    @property
    def roi(self):
        return (self.balance - self.initial_balance) / self.initial_balance

    @roi.setter
    def roi(self, value):
        self._roi = value

    def start_simulation(self):
        self.simulation_is_running = True

    def finish_simulation(self):
        self.simulation_is_running = False

    def buy(self, _symbol: str):
        if not self.simulation_is_running:
            print('simulação não está executando')
            return

        current_price = self.get_close_price_symbol_at(_symbol, self.index)

        if not self.open_position[0]:
            self.candlestick_count = 0
            self.profit = 0.0
            print(f'{_symbol} iniciando negociação de compra a {current_price}')
            self.open_position = ('buying', _symbol)
            self.starting_price = current_price
            self.num_buys += 1
        elif self.open_position[0] == 'selling' and self.open_position[1] == _symbol:
            self.close_position()
        elif self.open_position[0] == 'selling' and self.open_position[1] != _symbol:
            self.close_position()
            print(f'{_symbol} iniciando negociação de compra a {current_price}')
            self.open_position = ('buying', _symbol)
            self.starting_price = current_price
            self.num_buys += 1
        elif self.open_position[0] == 'buying' and self.open_position[1] == _symbol:
            print(f'operação negada. já tem uma negociação de compra pendente no mesmo ativo {_symbol}.')
        elif self.open_position[0] == 'buying' and self.open_position[1] != _symbol:
            self.close_position()
            print(f'{_symbol} iniciando negociação de compra a {current_price}')
            self.open_position = ('buying', _symbol)
            self.starting_price = current_price
            self.num_buys += 1

    def sell(self, _symbol: str):
        if not self.simulation_is_running:
            print('simulação não está executando')
            return

        current_price = self.get_close_price_symbol_at(_symbol, self.index)

        if not self.open_position[0]:
            self.candlestick_count = 0
            self.profit = 0.0
            print(f'{_symbol} iniciando negociação de compra a {current_price}')
            self.open_position = ('selling', _symbol)
            self.starting_price = current_price
            self.num_sells += 1
        elif self.open_position[0] == 'buying' and self.open_position[1] == _symbol:
            self.close_position()
        elif self.open_position[0] == 'buying' and self.open_position[1] != _symbol:
            self.close_position()
            print(f'{_symbol} iniciando negociação de venda a {current_price}')
            self.open_position = ('selling', _symbol)
            self.starting_price = current_price
            self.num_sells += 1
        elif self.open_position[0] == 'selling' and self.open_position[1] == _symbol:
            print(f'operação negada. já tem uma negociação de venda pendente no mesmo ativo {_symbol}.')
        elif self.open_position[0] == 'selling' and self.open_position[1] != _symbol:
            self.close_position()
            print(f'{_symbol} iniciando negociação de venda a {current_price}')
            self.open_position = ('selling', _symbol)
            self.starting_price = current_price
            self.num_sells += 1

    def update_profit(self):
        if not self.simulation_is_running:
            print('a simulação não está executando.')
            return

        if self.open_position[0] and self.open_position[1]:
            _symbol = self.open_position[1]
            current_price = self.get_close_price_symbol_at(_symbol, self.index)

            if self.open_position[0] == 'buying':
                self.profit = current_price - self.starting_price
            elif self.open_position[0] == 'selling':
                self.profit = self.starting_price - current_price

            self.profit = self.profit * self.contract_size * self.volume_operation
            self.equity = self.balance + self.profit

    def close_position(self):
        if not self.simulation_is_running:
            print('a simulação não está executando.')
            return

        if not self.open_position[0]:
            return

        _symbol = self.open_position[1]

        if self.open_position[0] == 'buying':
            print(f'{_symbol} fechando negociação de compra aberta. profit = {self.profit:.5f}')
            self.num_sells += 1
        elif self.open_position[0] == 'selling':
            print(f'{_symbol} fechando negociação de venda aberta. profit = {self.profit:.5f}')
            self.num_buys += 1

        self.open_position = ('', '')
        self.candlestick_count = 0

        if self.profit > 0:
            self.num_hits += 1
        else:
            self.num_misses += 1

        self.profit = 0.0
        self.balance = self.equity

    def interact_with_user(self) -> str:
        print('menu commands: ')
        cmd = input('quit(q) buy(b) sell(s) close(c) next(n) <-- ')
        cmd = cmd.upper().split()

        return_msg = 'continue'

        if cmd[0] == 'Q':
            return_msg = 'break'
        elif cmd[0] == 'B':
            if len(cmd) != 2:
                print('comando de compra inválido.')
                return_msg = 'continue'
            else:
                self.buy(cmd[1])
        elif cmd[0] == 'S':
            if len(cmd) != 2:
                print('comando de venda inválido.')
                return_msg = 'continue'
            else:
                self.sell(cmd[1])
        elif cmd[0] == 'C':
            self.close_position()
        elif cmd[0] == 'N':
            pass
        else:
            pass

        return return_msg

    def print_trade_stats(self):
        print(f'candlestick_count = {self.candlestick_count}, ', end='')
        print(f'open_position = {self.open_position}, ', end='')
        print(f'initial_balance = {self.initial_balance:.5f}, ', end='')
        print(f'balance = {self.balance:.5f}, ', end='')
        print(f'equity = {self.equity:.5f}')
        print(f'profit = {self.profit:.5f}, ', end='')
        print(f'num_hits = {self.num_hits}, ', end='')
        print(f'num_misses = {self.num_misses}, ', end='')
        print(f'num_trades = {self.num_trades}, ', end='')
        print(f'hit_rate = {self.hit_rate*100:.2f} %, ', end='')
        print(f'roi = {self.roi * 100:.5f} %')

    def get_close_price_symbol_at(self, _symbol: str, _index: int, use_scalers=True):
        """
        Obtém o preço de fechamento da vela do símbolo indicado no índice indicado.
        :param use_scalers:
        :param _symbol:
        :param _index:
        :return:
        """
        close_price_col = 5
        _symbol = f'{_symbol}_{self.timeframe}'
        _c = self.hist.arr[_symbol][_index, close_price_col]

        if use_scalers and self.scalers:
            trans: MinMaxScaler = self.scalers[_symbol]
            _c = denorm_close_price(_c, trans)

        return _c

    def print_symbols_close_price_at(self, _index, use_scalers=True):
        _list = []
        for _s in self.symbols:
            _c = self.get_close_price_symbol_at(_s, _index, use_scalers)
            _list.append(f'({_s} {_c:.5f})')
        _l = ' '.join(_list)
        print(_l)


def main():
    initial_deposit = 1000.0

    trader = TraderSimMulti(initial_deposit)
    trader.start_simulation()

    candlesticks_quantity = 50  # quantidade de velas que serão usadas na simulação

    for i in range(0, candlesticks_quantity):

        print(f'i = {i}')
        trader.index = i
        trader.print_symbols_close_price_at(i)

        trader.update_profit()

        if trader.profit < 0 and abs(trader.profit)/trader.balance >= trader.stop_loss:
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

        ret_msg = trader.interact_with_user()

        if ret_msg == 'break':
            print('o usuário decidiu encerrar a simulação.')
            trader.close_position()
            trader.finish_simulation()
            break

    print('\nresultados finais da simulação')
    trader.print_trade_stats()


# --------------------------------------------------------
if __name__ == '__main__':
    main()
