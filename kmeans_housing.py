import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

ARQUIVO_DADOS = 'HousingData.csv'
ARQUIVO_MODELO = 'modelo_kmeans_housing.pkl'
ARQUIVO_NORMALIZADOR = 'normalizador_housing.pkl'
ARQUIVO_FEATURES = 'features_housing.pkl'


def carregar_base(caminho):
    dados = pd.read_csv(caminho)

    if 'MEDV' in dados.columns:
        colunas_features = [col for col in dados.columns if col != 'MEDV']
    else:
        colunas_features = dados.columns.tolist()

    dados = dados[colunas_features].copy()
    dados = dados.apply(pd.to_numeric, errors='coerce')
    dados = dados.fillna(dados.median())
    return dados, colunas_features


def determinar_k_elbow(X, k_max=10):
    k_max = min(k_max, len(X) - 1)
    if k_max < 2:
        return 1

    inertias = []
    ks = list(range(1, k_max + 1))

    for k in ks:
        modelo = KMeans(n_clusters=k, random_state=42, n_init=20)
        modelo.fit(X)
        inertias.append(modelo.inertia_)

    p1 = np.array([ks[0], inertias[0]])
    p2 = np.array([ks[-1], inertias[-1]])

    distancias = []
    for k, inertia in zip(ks, inertias):
        p = np.array([k, inertia])
        numerador = abs((p2[1] - p1[1]) * p[0] - (p2[0] - p1[0]) * p[1] + p2[0] * p1[1] - p2[1] * p1[0])
        denominador = np.hypot(p2[1] - p1[1], p2[0] - p1[0])
        distancia = numerador / denominador if denominador != 0 else 0
        distancias.append(distancia)

    melhor_k = ks[int(np.argmax(distancias))]
    return melhor_k, ks, inertias


def main():
    dados, colunas_features = carregar_base(ARQUIVO_DADOS)

    normalizador = MinMaxScaler()
    dados_norm = pd.DataFrame(
        normalizador.fit_transform(dados[colunas_features]),
        columns=colunas_features,
        index=dados.index,
    )

    melhor_k, ks, inertias = determinar_k_elbow(dados_norm, k_max=10)

    modelo_kmeans = KMeans(n_clusters=melhor_k, random_state=42, n_init=20)
    modelo_kmeans.fit(dados_norm)

    with open(ARQUIVO_MODELO, 'wb') as arquivo:
        pickle.dump(modelo_kmeans, arquivo)

    with open(ARQUIVO_NORMALIZADOR, 'wb') as arquivo:
        pickle.dump(normalizador, arquivo)

    with open(ARQUIVO_FEATURES, 'wb') as arquivo:
        pickle.dump(colunas_features, arquivo)

    nova_residencia = dados.iloc[0].copy()
    nova_residencia_df = pd.DataFrame([nova_residencia.values], columns=colunas_features)
    nova_residencia_norm = pd.DataFrame(
        normalizador.transform(nova_residencia_df),
        columns=colunas_features,
        index=[0],
    )

    cluster_previsto = modelo_kmeans.predict(nova_residencia_norm)[0]
    centroide_norm = pd.Series(modelo_kmeans.cluster_centers_[cluster_previsto], index=colunas_features)
    centroide_desnormalizado = pd.Series(
        normalizador.inverse_transform([centroide_norm.values])[0],
        index=colunas_features,
    )

    print('Base carregada com sucesso:')
    print(dados.head())
    print('\nNúmero ótimo de clusters (método do cotovelo):', melhor_k)
    print('Inércias por k:', dict(zip(ks, inertias)))
    print('\nNova residência original:')
    print(nova_residencia_df)
    print('\nNova residência normalizada:')
    print(nova_residencia_norm)
    print('\nCluster previsto:', cluster_previsto)
    print('\nCentroide do cluster previsto (desnormalizado):')
    print(centroide_desnormalizado)


if __name__ == '__main__':
    main()
