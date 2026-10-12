import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
os.makedirs("Graficos_KNN", exist_ok=True)
data_set = pd.read_csv('Campeonato-brasileiro-estatisticas-2015-2023-completo.csv')

def resultado(row):
    if row['Vencedor'] == row['mandante']:
        return 'mandante'
    elif row['Vencedor'] == row['visitante']:
        return 'visitante'
    else:
        return 'empate'

data_set['resultado'] = data_set.apply(resultado, axis=1)


remover= [
    'ID', 'rodata', 'data', 'hora', 'mandante', 'visitante',
    'mandante_Placar', 'visitante_Placar', 'Vencedor', 'resultado'
]

X = data_set.drop(columns=remover)
y = data_set['resultado']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=14, stratify=y
)

Escalador = MinMaxScaler(feature_range=(0, 1))
x_train = Escalador.fit_transform(X_train)
x_test = Escalador.fit_transform(X_test)

scores_teste = []

for i in range(1, 40):

    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
    knn.fit(x_train, y_train)
    y_teste = knn.predict(x_test)
    scores_teste.append(knn.score(x_test, y_test))
    print(scores_teste[i-1])

plt.title('Acurácia do modelo de teste')
plt.xlabel('Valor de K')
plt.ylabel('Acurácia')
plt.plot(range(1,40),scores_teste)
plt.xticks(range(0,40,2))
plt.savefig(os.path.join("Graficos_KNN", "Acuracia_com_k_valores_diferentes_teste.png"), dpi=150, bbox_inches="tight")
plt.show()
plt.close()

scores_treinamento = []

for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
    knn.fit(x_train, y_train)
    y_teste = knn.predict(x_train)
    scores_treinamento.append(knn.score(x_train, y_train))
    print(scores_treinamento[i-1])

plt.title('Acurácia do modelo de treinamento')
plt.xlabel('Valor de K')
plt.ylabel('Acurácia')
plt.plot(range(1,20),scores_treinamento)
plt.xticks(range(0,20,1))
plt.savefig(os.path.join("Graficos_KNN", "Acuracia_com_k_valores_diferentes_treinamento.png"), dpi=150, bbox_inches="tight")
plt.show()
plt.close()