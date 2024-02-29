import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def processamento(file):
    dados = pd.read_csv(file)
    dados = dados.drop(['CUST_ID', 'TENURE'], axis=1)
    missing = dados.isna().sum()
    print(missing)
    dados.fillna(dados.median(), inplace=True)
    dadospca = dados.copy()
    
    return dados, dadospca


#Qual modelo de normalização usar? 
#Função StandardScaler() --> Útil para dados que estão em diferentes escalas e possuem diferentes unidades de medida.
#Função Normalizer() --> Útil para dados que possuem escalas similares, porém possuem unidades de medidas diferentes.
#Nesse caso, deve utilizar o Normalizer().


def normalizacao(dados):
    values = Normalizer().fit_transform(dados)
    return values



#Métrica de precisão ----------------------- Inetria(WCSS)
def calculate_inertia(X):
    inertia_values = []

# Testar diferentes números de clusters (K) para o K-Means
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        inertia = kmeans.inertia_
        inertia_values.append(inertia)
    print('Métrica de Precisão ------ Inertia: ',inertia_values)

# Plotar um gráfico do valor de Inertia em função do número de clusters (K)
    plt.plot(range(1, 11), inertia_values, marker='o')
    plt.title('Gráfico de Inertia em função do número de clusters (K)')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.show()
    
    return inertia_values


def optimal_number_of_clusters(inertia_values):
    x1, y1 = 2, inertia_values[0]
    x2, y2 = 11, inertia_values[len(inertia_values)-1]

    distances = []
    for i in range(len(inertia_values)):
        x0 = i+2
        y0 = inertia_values[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = (((y2 - y1)**2 + (x2 - x1)**2)**0.5)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2


def clustering(number_optimal, values, dados, dadospca):
    kmeans = KMeans(n_clusters=number_optimal, n_init=10, max_iter=300) # Definindo o número de clusters, irá variar de 0 até 4.
    y_pred = kmeans.fit_predict(values) #Previsão da segmentação de mercado
    y_pred = pd.DataFrame(y_pred)
    dados['Cluster'] = y_pred
    dadospca['Cluster'] = y_pred

    return kmeans, dados['Cluster'], dados, dadospca

def pca(dados):
    colunas = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
    dados_number = dados[colunas]
    #Padronização da base de dados
    number_columns = dados_number.columns
    standard = StandardScaler()
    dados_number = standard.fit_transform(dados_number)
    df_dados_number = pd.DataFrame(dados_number, columns=number_columns)

    #Construindo o PCA com todas as variavéis numéricas
    n_fatores = df_dados_number.shape[1]
    pca = PCA(n_components=n_fatores)
    pca.fit(df_dados_number) #PCA feito, agora, Análise de fatores
    components = pca.components_
    
    
    print('COEFICIENTES DA COMBINAÇÃO LINEAR:')
    print(components)
    print('')
    
    #Entrega a porcentagem de variância explicada por cada um dos fatores gerados pela PCA
    explaned_variance_ratio = pca.explained_variance_ratio_
    print('Autovetores:',explaned_variance_ratio) 
    print('')
    #Definindo nome para cada um dos fatores
    fatores = [f'F{i+1}' for i in range(n_fatores)]



    fig = plt.figure(figsize= (10, 5))
    plt.plot(explaned_variance_ratio, 'ro-', linewidth=3)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eingevalue (Autovalor)')
    plt.show()

    df_components = pd.DataFrame(components, columns=number_columns, index = [f'Autovetor {i+1}' for i in range(n_fatores)])
    df_components


    variancia_acumulada = [sum(explaned_variance_ratio[0:i+1]) for i in range(n_fatores)]
    variancia_acumulada = np.round(variancia_acumulada, 2)
    print('Variância Acumulada:', variancia_acumulada) #Autovalores acumulados


    fig = plt.figure(figsize= (10, 5))
    plt.plot(variancia_acumulada, 'ro-', linewidth=3)
    plt.title('Porcentagem da variancia acumulada')
    plt.xlabel('Principal Component')
    plt.ylabel('Eingevalue (Autovalor) Acumulada')
    plt.show()


    #pca.explanded_variance_ --> Representa a quantidade de variáveis por cada um dos fatores.
    #O valor mais exato pode ser explicado, multplicando: pca.explaned_variance_ratio pela quantidade de fatores.
    autovalores = pca.explained_variance_ratio_ * n_fatores
    print('proximação da Quantidade de Variáveis por cada um dos fatores:', autovalores)
    #Se realizar a soma, o segundo método é oq mais se aproxima de 9.

    fatores_selecionados = ['Fator selecionado' if autovalor > 1 else 'Fator não selecionado' for autovalor in autovalores]

    fig = plt.figure(figsize= (10, 5))
    plt.plot(autovalores, 'ro-', linewidth=3)
    plt.title('Scree Plot - Autovalores multplicados por 9')
    plt.xlabel('Componentes')
    plt.ylabel('Autovalor')
    plt.show()



    raiz_autovalores = np.sqrt(autovalores)
    print('Raiz Autovalores:',raiz_autovalores)

    cargas_fatoriais = pd.DataFrame(components.T * raiz_autovalores, columns=fatores, index = number_columns)
    print('Cargas Fatoriais:')
    print(cargas_fatoriais)

    fig = plt.figure(figsize=(10,5))
    plt.scatter(x=cargas_fatoriais['F1'], y=cargas_fatoriais['F2'])
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.show()

    #Pode-se observar que as variáveis que tem maior carga fatoriais e são relevantes para o rankeamento são as: Renda, Quota, Escolaridade e Idade.
    #Todas elas apresenta

    #Reduzindo a dimensionalidade para 4 variáveis
    #Método Aula professora USP

    pca = PCA(n_components=3)
    pca.fit(df_dados_number)
    components_principais = pca.components_
    print(components_principais)


    components_scores = []
    for i in range(3):
        scores = pca.transform(df_dados_number)[:,i]
        components_scores.append(scores)

    components_scores = pd.DataFrame(components_scores).T
    print(components_scores)

    dados['scoresCP1'] = components_scores[0]
    dados['scoresCP2'] = components_scores[1]
    dados['scoresCP3'] = components_scores[2]
    
    #Scores --> indicam os valores de relação de uma variável com a componente principal em questão


    dados['Ranking'] = dados['scoresCP1'] * explaned_variance_ratio[0] + dados['scoresCP2'] * explaned_variance_ratio[1]
    print(dados)

    filtro_scorecp1 = dados.sort_values(by='scoresCP1', ascending=False)
    filtro_scorecp1 = filtro_scorecp1.drop(['scoresCP3'], axis=1)
    colunas = ['BALANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'scoresCP1', 'scoresCP2', 'Ranking', 'Cluster']
    return print(filtro_scorecp1[colunas])