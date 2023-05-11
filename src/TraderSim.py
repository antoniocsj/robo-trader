from Hist import Hist


class TraderSim:
    def __init__(self, symbol: str, timeframe: str, balance: float) -> None:
        self.symbol = symbol  # financial asset, security or contract etc.
        self.timeframe = timeframe
        self.hist = Hist()
        self.open_position = None
        self.candlestick_count = 0
        self.symbol = ''
        self.timeframe = ''
        self.num_hits = 0  # número de acertos
        self.num_misses = 0  # número de erros
        self.num_buys = 0  # número de negociações de compra
        self.num_sells = 0  # número de negociações de venda
        self.num_trades = 0  # número de negociações de compra ou venda
        self.starting_price = 0.0
        self.final_price = 0.0
        self.balance = balance  # saldo
        self.equity = 0.0  # patrimônio líquido
        self.profit = 0.0  # lucro (ou prejuízo) na negociação
        self.hist.get_hist_data(symbol, timeframe)

    def buy(self, current_price):
        if self.open_position is None:
            self.candlestick_count = 0
            print(f'iniciando operação de compra a {current_price}')
            self.open_position = 'buying'
            self.starting_price = current_price
            self.num_buys += 1
        elif self.open_position == 'selling':
            self.candlestick_count = 0
            self.close_position(current_price)
            print(f'iniciando operação de compra a {current_price}')
            self.open_position = 'buying'
            self.starting_price = current_price
            self.num_buys += 1
        elif self.open_position == 'buying':
            print('proibido comprar com uma compra já aberta')

    def sell(self, current_price):
        if self.open_position is None:
            self.candlestick_count = 0
            print(f'iniciando operação de venda a {current_price}')
            self.open_position = 'selling'
            self.starting_price = current_price
            self.num_sells += 1
        elif self.open_position == 'buying':
            self.candlestick_count = 0
            self.close_position(current_price)
            print(f'iniciando operação de venda a {current_price}')
            self.open_position = 'selling'
            self.starting_price = current_price
            self.num_sells += 1
        elif self.open_position == 'selling':
            print('proibido vender com uma venda já aberta')

    def close_position(self, current_price):
        self.final_price = current_price

        if self.open_position == 'buying':
            print('fechando operação de compra aberta')
            self.num_sells += 1
            self.profit = self.final_price - self.starting_price
            print(f'profit = {self.profit}')
        elif self.open_position == 'selling':
            print('fechando operação de venda aberta')
            self.num_buys += 1
            self.profit = self.starting_price - self.final_price
            print(f'profit = {self.profit}')

        self.balance += self.profit
        self.open_position = None

        if self.profit > 0:
            self.num_hits += 1
        else:
            self.num_misses += 1


def main():
    symbol = 'XAUUSD'
    timeframe = 'H1'

    hist = Hist()
    hist.get_hist_data(symbol, timeframe)
    hist.print_hist()

    # teste do TraderSim
    trader = TraderSim(symbol, timeframe, 1000.0)

    # fazer um sistema interativo, no qual o usuário pode operar como se estivesse no MT5.

    previous_price = hist.arr[0, 5]
    n = 10
    for i in range(1, n):
        current_price = hist.arr[i, 5]
        print(f'i = {i}')
        print(f'current_price = {current_price}')

        if current_price > previous_price:
            trader.buy(current_price)

        if current_price < previous_price:
            trader.sell(current_price)

        # fecha a posição quando acabarem as bnovas barras (velas ou candlesticks)
        if i == n - 1:
            trader.close_position(current_price)
            trader.candlestick_count = 0

        print(f'candlestick = {trader.candlestick_count}')
        print(f'open_position = {trader.open_position}')
        print(f'balance = {trader.balance}')
        print(f'equity = {trader.equity}')
        print(f'profit = {trader.profit}')
        print(f'num_hits = {trader.num_hits}')
        print(f'num_misses = {trader.num_misses}')

        trader.candlestick_count += 1
        previous_price = current_price


# --------------------------------------------------------
if __name__ == '__main__':
    main()
