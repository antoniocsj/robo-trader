from Hist import Hist


class TraderSim:
    def __init__(self, symbol: str, timeframe: str, balance: float) -> None:
        self.symbol = symbol  # financial asset, security or contract etc.
        self.timeframe = timeframe
        self.hist = Hist()
        self.open_position = None
        self.candlestick_count = 0  # contagem de velas desde a abertura da posição
        self.symbol = ''
        self.timeframe = ''
        self.simulation_is_running = False
        self.num_hits = 0  # número de acertos
        self.num_misses = 0  # número de erros
        self.num_buys = 0  # número de negociações de compra
        self.num_sells = 0  # número de negociações de venda
        self.num_trades = 0  # número de negociações de compra ou venda
        self.current_price = 0.0  # preço atual ou o preço de fechamento atual
        self.previous_price = 0.0  # preço anterior ou o preço de fechamento da vela anterior
        self.starting_price = 0.0
        self.final_price = 0.0
        self.balance = balance  # saldo
        self.equity = 0.0  # patrimônio líquido
        self.profit = 0.0  # lucro (ou prejuízo) na negociação
        self.hist.get_hist_data(symbol, timeframe)

    def start_simulation(self):
        self.simulation_is_running = True

    def finish_simulation(self):
        self.simulation_is_running = False

    def buy(self):
        if not self.simulation_is_running:
            print('simulação não está executando')
            return

        if self.open_position is None:
            self.candlestick_count = 0
            self.profit = 0.0
            print(f'iniciando operação de compra a {self.current_price}')
            self.open_position = 'buying'
            self.starting_price = self.current_price
            self.num_buys += 1
        elif self.open_position == 'selling':
            self.candlestick_count = 0
            self.profit = 0.0
            self.close_position()
            print(f'iniciando operação de compra a {self.current_price}')
            self.open_position = 'buying'
            self.starting_price = self.current_price
            self.num_buys += 1
        elif self.open_position == 'buying':
            print('proibido comprar com uma compra já aberta')

    def sell(self):
        if not self.simulation_is_running:
            print('simulação não está executando')
            return

        if self.open_position is None:
            self.candlestick_count = 0
            self.profit = 0.0
            print(f'iniciando operação de venda a {self.current_price}')
            self.open_position = 'selling'
            self.starting_price = self.current_price
            self.num_sells += 1
        elif self.open_position == 'buying':
            self.candlestick_count = 0
            self.profit = 0.0
            self.close_position()
            print(f'iniciando operação de venda a {self.current_price}')
            self.open_position = 'selling'
            self.starting_price = self.current_price
            self.num_sells += 1
        elif self.open_position == 'selling':
            print('proibido vender com uma venda já aberta')

    def update_profit(self):
        if not self.simulation_is_running:
            print('simulação não está executando')
            return

        if self.open_position == 'buying':
            self.profit = self.current_price - self.starting_price
        elif self.open_position == 'selling':
            self.profit = self.starting_price - self.current_price

        self.equity = self.balance + self.profit

    def close_position(self):
        if not self.simulation_is_running:
            print('simulação não está executando')
            return

        if self.open_position is None:
            return

        if self.open_position == 'buying':
            print('fechando operação de compra aberta')
            self.num_sells += 1
        elif self.open_position == 'selling':
            print('fechando operação de venda aberta')
            self.num_buys += 1

        self.update_profit()
        self.open_position = None

        if self.profit > 0:
            self.num_hits += 1
        else:
            self.num_misses += 1

        self.profit = 0.0
        self.balance = self.equity

    def interact_with_user(self) -> str:
        print('menu commands: ')
        cmd = input('quit(q) buy(b) sell(s) close(c) next(n) <-- ')
        cmd = cmd.lower()

        return_msg = 'continue'

        if cmd == 'q':
            return_msg = 'break'
        elif cmd == 'b':
            self.buy()
        elif cmd == 's':
            self.sell()
        elif cmd == 'c':
            self.close_position()
        elif cmd == 'n':
            pass
        else:
            pass

        return return_msg

    def print_trade_stats(self):
        print(f'candlestick_count = {self.candlestick_count}, ', end='')
        print(f'open_position = {self.open_position}, ', end='')
        print(f'balance = {self.balance:.2f}, ', end='')
        print(f'equity = {self.equity:.2f}')
        print(f'profit = {self.profit:.2f}, ', end='')
        print(f'num_hits = {self.num_hits}, ', end='')
        print(f'num_misses = {self.num_misses}')


def main():
    symbol = 'XAUUSD'
    timeframe = 'H1'

    hist = Hist()
    hist.get_hist_data(symbol, timeframe)
    # hist.print_hist()

    # teste do TraderSim
    trader = TraderSim(symbol, timeframe, 1000.0)
    trader.start_simulation()

    # fazer um sistema interativo, no qual o usuário pode operar como se estivesse no MT5.
    close_price_col = 5
    trader.previous_price = hist.arr[0, close_price_col]
    candlesticks_quantity = 50  # quantidade de velas que serão usadas na simulação

    for i in range(0, candlesticks_quantity):
        trader.current_price = hist.arr[i, close_price_col]
        print(f'i = {i}, ', end='')
        print(f'OHLCV = {hist.arr[i]}, ', end='')
        print(f'current_price = {trader.current_price:.2f}, ', end='')
        print(f'price_delta = {trader.current_price-trader.previous_price:.2f}')

        trader.update_profit()

        # fecha a posição quando acabarem as bnovas barras (velas ou candlesticks)
        if i == candlesticks_quantity - 1:
            trader.close_position()
            trader.candlestick_count = 0
            trader.finish_simulation()
            print('última vela concluída. simulação chegou ao fim.')

        trader.print_trade_stats()

        if trader.open_position:
            trader.candlestick_count += 1
        else:
            trader.candlestick_count = 0

        trader.previous_price = trader.current_price
        ret_msg = trader.interact_with_user()

        if ret_msg == 'break':
            print('usuário decidiu encerrar a simulação')
            trader.close_position()
            trader.finish_simulation()
            break


# --------------------------------------------------------
if __name__ == '__main__':
    main()
