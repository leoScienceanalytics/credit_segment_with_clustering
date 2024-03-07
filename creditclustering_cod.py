import pandas as pd
from sklearn import metrics
from creditclustering_functions import processamento, normalizacao, pca 
from creditclustering_functions import calculate_inertia, optimal_number_of_clusters, clustering
import matplotlib.pyplot as plt


dados, dadospca = processamento('creditcustomersegmentation.csv')

values = normalizacao(dados)


values = pd.DataFrame(values)
sum_of_squares = calculate_inertia(values)
number_optimal = optimal_number_of_clusters(sum_of_squares)
print('Número ótimo de clusters:',number_optimal)
 
kmeans, cluster, dados, dadospca = clustering(number_optimal, values, dados, dadospca)
print(dados)


#Métrica do Silhoutette
labels = kmeans.labels_
silhouette = metrics.silhouette_score(values, labels, metric='euclidean')
print('Silhouette:',silhouette)

#Métrica Davies-Bouldin
dbs = metrics.davies_bouldin_score(values, labels)
print('Davies-Bouldin Index:',dbs)

#Métrica Calinski
calinski = metrics.calinski_harabasz_score(values, labels)
print('Calinski:',calinski)

#Análise dos Clusters
dados.groupby('Cluster').describe()
centroids = kmeans.cluster_centers_
print(centroids)
max = len(centroids[0])
for i in range(max):
    print(dados.columns.values[i], '\n{:.4f}'.format(centroids[:, i].var()))


colunas = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
description = dados.groupby('Cluster')[colunas]
n_clients = description.size()
description = description.mean()
description['n_clients'] = n_clients
print(description)


print(dados.groupby('Cluster')['PRC_FULL_PAYMENT'].describe())

dados, score = pca(dadospca)
print('Rankeamento dos Clusters')
print(score)


cluster_counts = dados['Cluster'].value_counts()
plt.bar(cluster_counts.index, cluster_counts)
plt.title('Quantidade por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Quantidade')
plt.show()