import pandas as pd
import numpy as np
from itertools import groupby
from datetime import datetime
import mplcyberpunk 
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdate
import seaborn as sns
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import os


class ReportResult():

    def __init__(self, dfTrades, dfCarteiras, caminhoImagens):

        self.carteiras = dfCarteiras
        self.dfTrades = dfTrades
        self.caminhoImagens = caminhoImagens

        self.dfTrades['data'] = pd.to_datetime(self.dfTrades['data'])
        self.dfTrades['dinheiro'] = self.dfTrades['dinheiro'].astype(float)
        self.dfTrades['retorno'] = self.dfTrades['retorno'].astype(float)

        ibov = pd.read_parquet('ibov.parquet')
        ibov['data'] = pd.to_datetime(ibov['data'])
        ibov['retorno'] = ibov['fechamento'].pct_change()
        self.ibov = ibov

        cdi = pd.read_parquet('cdi.parquet')
        cdi['data'] = pd.to_datetime(cdi['data'])
        cdi['cota'] = (1 + cdi['retorno']).cumprod() - 1
        self.cdi = cdi

        self.ibov = self.ibov[(self.ibov['data'] >= self.dfTrades['data'].iloc[0]) & (self.ibov['data'] <= self.dfTrades['data'].iloc[-1])]
        self.cdi = self.cdi[(self.cdi['data'] >= self.dfTrades['data'].iloc[0]) & (self.cdi['data'] <= self.dfTrades['data'].iloc[-1])]

        plt.style.use('cyberpunk')

        self.make_report()

    def make_report(self):

        self.periodo_backtest()
        self.risco_retorno()
        self.turnover_carteira()
        self.drawdown()
        self.estatisticas_de_trade()
        self.grafico_retorno_acumulado()
        self.underwater()
        self.retorno_mes_a_mes()
        self.retorno_ano_a_ano()

        print('')
        print('-------------------------------------')
        print('  ESTATISTICAS DE RETORNO E RISCO')
        print('-------------------------------------')
        print(f'Retorno Acumulado modelo: {self.retornoAcumModelo * 100:.2f}%')
        print(f'Retorno Acumulado CDI: {self.retornoAcumCdi * 100:.2f}%')
        print(f'Retorno Acumulado IBOV: {self.retornoAcumIbov * 100:.2f}%')
        print(f'Retorno Anual Modelo: {self.retornoAnoModelo * 100:.2f}%')
        print(f'Volatilidade Ultimo Ano: {self.volUltimoAno * 100:.2f}%')
        print(f'Indice Sharpe: {self.sharpe:.2f}')
        print(f'Turno over carteiras: {self.turnoverMedio * 100:.2f}%')
        print(f'Drawdown maximo: {self.MaxDrawdown * 100:.2f}%')
        print('-------------------------------------')
        print('       ESTATISTICAS DE TRADE')
        print('-------------------------------------')
        print(f'Numero de carteiras: {self.numeroTrades}')
        print(f'% Operacoes vencedoras: {self.operacoesVencedoras * 100:.2f}%')
        print(f'% Operacoes perdedoras: {self.operacoesPerdedoras* 100:.2f}%')
        print(f'Media de ganhos: {self.mediaGanhos* 100:.2f}%')
        print(f'Media de perdas: {self.mediaPerdas* 100:.2f}%')
        print(f'Expec. matematica trade: {self.expectativaMatematica* 100:.2f}%')
        print(f'Mario sequecia de vitorias: {self.maiorSequenciaVitorias}')
        print(f'Mario sequencia de derrotas: {self.maiorSequeciaDerrotas}')
        print(f'% meses carteira > IBOV: {self.percentualCartSuperaIbov * 100:.2f}%')
        print('-------------------------------------')

    def periodo_backtest(self):

        self.diaInical = self.dfTrades['data'].iat[0]
        self.diaFinal= self.dfTrades['data'].iat[-1]

        self.diasTotaisBacktest = len(self.dfTrades)

    def risco_retorno(self):
        
        self.retornoAcumModelo = self.dfTrades['dinheiro'].iat[-1]/self.dfTrades['dinheiro'].iat[0] - 1
        self.retornoAcumCdi = ((self.cdi['retorno'] + 1).cumprod() - 1).iat[-1]
        self.retornoAcumIbov = ((self.ibov['retorno'] + 1).cumprod() - 1).iat[-1]

        self.retornoAnoModelo = (1 + self.retornoAcumModelo) ** (252/self.diasTotaisBacktest) - 1
        self.retornoAnoCdi = (1 + self.retornoAcumCdi) ** (252/self.diasTotaisBacktest) - 1

        self.volUltimoAno = ((self.dfTrades['retorno'].iloc[-253:-1]).std()) * np.sqrt(252)
        self.volPeriodo = (self.dfTrades['retorno'].std()) * np.sqrt(252)

        self.sharpe = (self.retornoAnoModelo - self.retornoAnoCdi)/self.volPeriodo

        dia95 = int(self.diasTotaisBacktest * 0.05)
        self.varDiario = self.dfTrades['retorno'].sort_values(ascending= True).iat[dia95]

    def turnover_carteira(self):

        turnover = self.carteiras[['ticker']].copy()
        turnover['contador'] = 1

        turnover = turnover.groupby(['data', 'ticker']).count().unstack(fill_value=0)

        turnoverAnterior = turnover.shift()
        entrou = (turnover - turnoverAnterior) > 0
        saiu = (turnover - turnoverAnterior) < 0

        totalAnterior = turnoverAnterior.sum(axis=1)
        totalAtual = turnover.sum(axis=1)
        turnover = (saiu.sum(axis=1) + entrou.sum(axis=1)) / (totalAnterior + totalAtual)

        self.turnoverMedio = turnover.mean()

    def drawdown(self):

        df = self.dfTrades

        df['maximo'] = df['dinheiro'].rolling(len(self.dfTrades), min_periods=1).max()
        df['drawdown'] = (df['dinheiro'])/df['maximo'] - 1
        df['drawdown_max'] = df['drawdown'].rolling(len(self.dfTrades), min_periods=1).min()

        self.drawdowns = df['drawdown']
        self.drawdowns.index = df['data']

        self.MaxDrawdown = min(df['drawdown_max'])


    def estatisticas_de_trade(self):

        self.numeroTrades = self.dfTrades['numero_trade'].max()

        self.dfTrades['retorno_plus_one'] = self.dfTrades['retorno'] + 1

        self.dfTrades['retorno_por_trade'] = self.dfTrades.groupby('numero_trade')['retorno_plus_one'].cumprod() - 1

        tradesAcum = (self.dfTrades.groupby(['numero_trade']).tail(1))['retorno_por_trade']

        trades = tradesAcum.dropna().unique()

        self.operacoesVencedoras = np.sum(np.where(trades > 0, True, False))/self.numeroTrades

        self.operacoesPerdedoras = 1 - self.operacoesVencedoras

        self.mediaGanhos = trades[trades > 0].mean()

        self.mediaPerdas = trades[trades < 0].mean()

        self.expectativaMatematica = (self.operacoesVencedoras * self.mediaGanhos) - (self.operacoesPerdedoras * abs(self.mediaPerdas))

        self.maiorSequenciaVitorias = len(max((list(g) if k else [] for k, g in groupby(trades.tolist(), key=lambda i: i > 0)), key=len))
        self.maiorSequeciaDerrotas = len(max((list(g) if k else [] for k, g in groupby(trades.tolist(), key=lambda i: i < 0)), key=len))

        ibov = self.ibov.copy()
        ibov.columns = ['data', 'fechamento', 'retorno_ibov']

        dfTradesIbov = pd.merge(self.dfTrades, ibov, on='data', how='inner')
        
        dfTradesIbov['retorno_acum_ibov'] = (dfTradesIbov.groupby('numero_trade', group_keys=False)['retorno_ibov'].apply(lambda x: (1 + x).cumprod() - 1))
        dfTradesIbov['retorno_por_trade'] = (dfTradesIbov.groupby('numero_trade', group_keys=False)['retorno'].apply(lambda x: (1 + x).cumprod() - 1))

        retornoIbovPorTrade = dfTradesIbov.groupby('numero_trade')['retorno_acum_ibov'].last()
        retornoPorTrade = dfTradesIbov.groupby('numero_trade')['retorno_por_trade'].last()

        superouTrades = retornoPorTrade > retornoIbovPorTrade
        self.percentualCartSuperaIbov = superouTrades.mean()

    def grafico_retorno_acumulado(self):

        fig, ax = plt.subplots(figsize = (7,4))

        rentModelo = (self.dfTrades['retorno'] + 1).cumprod() - 1
        rentCdi = (self.cdi['retorno'] + 1).cumprod() - 1
        rentIbov = (self.ibov['retorno'] + 1).cumprod() - 1

        ax.plot(self.cdi['data'].values, rentCdi.values, label = 'CDI')
        ax.plot(self.ibov['data'].values, rentIbov.values, label = 'IBOV')
        ax.plot(self.dfTrades['data'].values, rentModelo.values, label = 'MODELO')

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

        plt.legend()
        plt.title('Retorno acumulado')
        ax.grid(False)

        plt.savefig(f'{self.caminhoImagens}/retorno_acumulado.png', dpi= 300)

        plt.close()

    def retorno_mes_a_mes(self):

        mensal = self.dfTrades
        mensal = self.dfTrades.set_index('data')
        mensal = mensal.resample('M').last()
        
        rentMes = mensal['dinheiro'].pct_change()
        rentMes = rentMes.to_frame()
        rentMes.columns = ['rent']
        rentMes['mes'] = rentMes.index.month_name()
        rentMes['mes'] = rentMes['mes'].apply(lambda x: x[0:3])
        rentMes['ano'] = rentMes.index.year
        rentMes = rentMes.pivot(index= 'ano', columns= 'mes', values= 'rent')
        rentMes = rentMes[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
        rentMes = rentMes.fillna(0)
        rentMes = rentMes * 100

        plt.style.use('default')

        fig = plt.figure(figsize= (8.75, 4))

        ax = sns.heatmap(rentMes, cmap= "YlGnBu", annot= True)

        plt.title('Retorno mês a mês')

        for t in ax.texts:
            t.set_text(t.get_text() + '%')

        plt.savefig(f'{self.caminhoImagens}/grafico_mes.png', dpi=300)

        plt.close

    def retorno_ano_a_ano(self):

        listaDfs = [self.dfTrades, self.cdi, self.ibov]
        nomes = ['MODELO', 'CDI', 'IBOV']

        dfTradesCopy = self.dfTrades.copy()
        dfTradesCopy = dfTradesCopy.set_index('data')

        self.dfAnual = pd.DataFrame(columns=['MODELO', 'CDI', 'IBOV'], index= np.unique(dfTradesCopy.index.year).tolist())

        for i, df in enumerate(listaDfs):

            self.transformando_em_anual(df, nomes[i])

        fig = plt.figure(figsize= (8.75, 4))

        fig.patch.set_facecolor('white')

        ax = sns.heatmap(self.dfAnual, cmap= 'YlGnBu', annot= True, fmt= '.3g')
        plt.title('Retorno ano a ano')

        for t in ax.texts:

            t.set_text(t.get_text() + '%')

        plt.savefig(f'{self.caminhoImagens}/grafico_ano.png', dpi = 300)

        plt.close()

    def transformando_em_anual(self, df, nome):

        dfRentAnual = df
        dfRentAnual = dfRentAnual.set_index('data')
        dfRentAnual['cota'] = dfRentAnual['retorno'] + 1
        dfRentAnual['ano'] = dfRentAnual.index.year
        dfRentAnual['retorno_anual'] = dfRentAnual.groupby('ano')['cota'].cumprod() - 1
        dfRentAnual = (dfRentAnual.groupby('ano').tail(1))['retorno_anual']

        self.dfAnual[nome] = dfRentAnual.values * 100

    def underwater(self):

        fig, ax = plt.subplots(figsize= (7, 4.5))

        ax.plot(self.drawdowns.index, self.drawdowns)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

        if self.diasTotaisBacktest > 1000:

            ax.xaxis.set_major_locator(mdate.YearLocator(2))

        else:

            ax.xaxis.set_major_locator(mdate.YearLocator(1))
        
        ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))
        plt.title('underwater')
        ax.grid(False)

        plt.savefig(f'{self.caminhoImagens}/grafico_underwater.png', dpi= 300)

        plt.close()

if __name__ == '__main__':
    
    os.chdir('base_dados')
    trades = pd.read_parquet('trades.parquet')
    carteiras = pd.read_parquet('carteiras.parquet')
    carteiras = carteiras.set_index('data')
    carteiras.index = pd.to_datetime(carteiras.index)

    results = ReportResult(dfTrades = trades, dfCarteiras = carteiras, caminhoImagens= r'C:\Users\Caio\Documents\dev\github\backtest_value_investing\imagens')

    # results.periodo_backtest()
    # results.risco_retorno()
    # results.turnover_carteira()
    # results.drawdown()
    # results.estatisticas_de_trade()
    # results.grafico_retorno_acumulado()
    # results.underwater()
    # results.retorno_mes_a_mes()
    # results.retorno_ano_a_ano()

    # print(results.retornoAcumModelo)
    # print(results.retornoAcumIbov)
    # print(results.retornoAcumCdi)
    # print(results.retornoAnoModelo)
    # print(results.retornoAnoCdi)
    # print(results.volUltimoAno)
    # print(results.volPeriodo)
    # print(results.sharpe)
    # print(results.varDiario)

    # print(results.turnoverMedio)

    # print(results.MaxDrawdown)

    # print(results.operacoesVencedoras)
    # print(results.operacoesPerdedoras)
    # print(results.mediaGanhos)
    # print(results.mediaPerdas)
    # print(results.expectativaMatematica)
    # print(results.maiorSequenciaVitorias)
    # print(results.maiorSequeciaDerrotas)
    # print(results.percentualCartSuperaIbov)