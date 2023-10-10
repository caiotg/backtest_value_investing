import os
import dotenv
import requests
import pandas as pd
import urllib.request

class LoadData():

    def __init__(self, caminhoDados):

        dotenv.load_dotenv()

        self.chaveApi = os.getenv('API_FINTZ')
        self.headers = {'accept': 'application/json',
                        'X-API-Key': self.chaveApi}
        
        os.chdir(caminhoDados)

    def cdi(self):

        response = requests.get('https://api.fintz.com.br/taxas/historico?codigo=12&dataInicio=2000-01-01&ordem=ASC',headers=self.headers)

        cdi = pd.DataFrame(response.json())
        cdi = cdi.drop(['dataFim', 'nome'], axis=1)
        cdi.columns = ['data', 'retorno']
        cdi['retorno'] = cdi['retorno']/100

        cdi.to_parquet('cdi.parquet', index= False)

    def ibov(self):

        response = requests.get('https://api.fintz.com.br/indices/historico?indice=IBOV&dataInicio=2000-01-01', headers=self.headers)

        ibov = pd.DataFrame(response.json())
        ibov = ibov.sort_values('data', ascending=True)
        ibov.columns = ['indice', 'data', 'fechamento']
        ibov = ibov.drop('indice', axis=1)

        ibov.to_parquet('ibov.parquet', index=False)

    def pegar_cotacoes(self):

        response = requests.get('https://api.fintz.com.br/bolsa/b3/avista/cotacoes/historico/arquivos?classe=ACOES&preencher=true', headers=self.headers)

        linkDownload = (response.json())['link']
        urllib.request.urlretrieve(linkDownload, f'cotacoes.parquet')

        cotacoes = pd.read_parquet('cotacoes.parquet')

        colunaParaAjustar = ['preco_abertura', 'preco_maximo', 'preco_medio', 'preco_minimo']

        for coluna in colunaParaAjustar:

            cotacoes[f'{coluna}_ajustado'] = cotacoes[coluna] * cotacoes['fator_ajuste']
        
        cotacoes['preco_fechamento_ajustado'] = cotacoes.groupby('ticker')['preco_fechamento_ajustado'].transform('ffill')
        cotacoes = cotacoes.sort_values('data', ascending=True)

        cotacoes.to_parquet('cotacoes.parquet', index=False)

    def indicadores(self, nomeDado = ''):

        try:

            response = requests.get(f'https://api.fintz.com.br/bolsa/b3/tm/indicadores/arquivos?indicador={nomeDado}', headers=self.headers)

        except:

            print('Indicador não encontrada!')
            exit()
        
        linkDownload = (response.json())['link']
        urllib.request.urlretrieve(linkDownload, f'{nomeDado}.parquet')

    def volume_mediano(self):

        cotacoes = pd.read_parquet('cotacoes.parquet')
        cotacoes['data'] = pd.to_datetime(cotacoes['data']).dt.date

        cotacoes = cotacoes[['data', 'ticker', 'volume_negociado']]
        cotacoes['volume_negociado'] = cotacoes.groupby('ticker')['volume_negociado'].fillna(0)
        cotacoes['valor'] = cotacoes.groupby('ticker')['volume_negociado'].rolling(21).median().reset_index(0,drop=True)
        cotacoes = cotacoes.dropna()

        valor = cotacoes[['data', 'ticker', 'valor']]

        valor.to_parquet('volume_mediano.parquet', index= False)


if __name__ == '__main__':

    dados = LoadData(caminhoDados= r'C:\Users\Caio\Documents\dev\github\backtest_value_investing\base_dados')


    dados.cdi()
    dados.ibov()
    dados.pegar_cotacoes()

    listaIndicadores = ['EBIT_EV','ValorDeMercado']

    for indicador in listaIndicadores:

        print(indicador)
        dados.indicadores(nomeDado=indicador)

    dados.volume_mediano()


