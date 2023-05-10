from Hist import Hist


class TraderSim:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.hist = Hist()
        self.posicao = None
        self.bar = 0
        self.symbol = ''
        self.timeframe = ''
        self.num_acertos = 0
        self.num_erros = 0
        self.num_compras = 0
        self.num_vendas = 0
        self.num_operacoes = 0
        self.preco_comeco = 0.0
        self.preco_fim = 0.0
        self.saldo_total = 0.0
        self.hist.obter_historico(symbol, timeframe)

    def comprar(self, preco_atual):
        if self.posicao is None:
            self.bar = 0
            print(f'iniciando operação de compra a {preco_atual}')
            self.posicao = 'comprado'
            self.preco_comeco = preco_atual
            self.num_compras += 1
        elif self.posicao == 'vendido':
            self.bar = 0
            self.fechar_posicao(preco_atual)
            print(f'iniciando operação de compra a {preco_atual}')
            self.posicao = 'comprado'
            self.preco_comeco = preco_atual
            self.num_compras += 1
        elif self.posicao == 'comprado':
            print('proibido comprar com uma compra pendente')

    def vender(self, preco_atual):
        if self.posicao is None:
            self.bar = 0
            print(f'iniciando operação de venda a {preco_atual}')
            self.posicao = 'vendido'
            self.preco_comeco = preco_atual
            self.num_vendas += 1
        elif self.posicao == 'comprado':
            self.bar = 0
            self.fechar_posicao(preco_atual)
            print(f'iniciando operação de venda a {preco_atual}')
            self.posicao = 'vendido'
            self.preco_comeco = preco_atual
            self.num_vendas += 1
        elif self.posicao == 'vendido':
            print('proibido vender com uma venda pendente')

    def fechar_posicao(self, preco_atual):
        self.preco_fim = preco_atual
        saldo_operacao = 0

        if self.posicao == 'comprado':
            print('fechando operação de compra pendente')
            self.num_vendas += 1
            saldo_operacao = self.preco_fim - self.preco_comeco
            print(f'saldo_operacao = {saldo_operacao}')
        elif self.posicao == 'vendido':
            print('fechando operação de venda pendende')
            self.num_compras += 1
            saldo_operacao = self.preco_comeco - self.preco_fim
            print(f'saldo_operacao = {saldo_operacao}')

        self.saldo_total += saldo_operacao
        self.posicao = None

        if saldo_operacao > 0:
            self.num_acertos += 1
        else:
            self.num_erros += 1


def main():
    symbol = 'XAUUSD'
    timeframe = 'H1'

    hist = Hist()
    hist.obter_historico(symbol, timeframe)
    hist.mostrar_historico()

    # teste do TraderSim
    trader = TraderSim(symbol, timeframe)

    preco_anterior = hist.arr[0, 5]
    n = 10
    for i in range(1, n):
        preco_atual = hist.arr[i, 5]
        print(f'i = {i}')
        print(f'preco_atual = {preco_atual}')

        if preco_atual > preco_anterior:
            trader.comprar(preco_atual)

        if preco_atual < preco_anterior:
            trader.vender(preco_atual)

        # fecha a posição quando acabar os novos valores
        if i == n - 1:
            trader.fechar_posicao(preco_atual)
            trader.bar = 0

        print(f'barra = {trader.bar}')
        print(f'posicao = {trader.posicao}')
        print(f'saldo_total = {trader.saldo_total}')
        print(f'num_acertos = {trader.num_acertos}')
        print(f'num_erros = {trader.num_erros}')

        trader.bar += 1
        preco_anterior = preco_atual


# --------------------------------------------------------
if __name__ == '__main__':
    main()
