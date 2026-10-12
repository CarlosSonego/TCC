import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
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

knn = KNeighborsClassifier(n_neighbors=38, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
knn.fit(x_train, y_train)

y_teste = knn.predict(x_test)
print(classification_report(y_test, y_teste))
print('Acurácia:',knn.score(x_test, y_test))

classes = ['empate', 'mandante', 'visitante']

fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_teste,
    labels=classes,
    display_labels=classes,
    cmap='Blues',
    ax=ax,
)
plt.title('Matriz de Confusão do modelo de teste')
plt.xlabel('Valor previsto pelo modelo')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig(os.path.join("Graficos_KNN", "Matriz_confusão_teste.png"), dpi=150, bbox_inches="tight")
plt.show()
plt.close()
y_treinamento = knn.predict(x_train)
print(classification_report(y_train, y_treinamento))
print('Acurácia:',knn.score(x_train, y_train))

figi, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay.from_predictions(
    y_train,
    y_treinamento,
    labels=classes,
    display_labels=classes,
    cmap='Blues',
    ax=ax,
)
plt.title('Matriz de Confusão do modelo de teste')
plt.xlabel('Valor previsto pelo modelo')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig(os.path.join("Graficos_KNN", "Matriz_confusão_treinamento.png"), dpi=150, bbox_inches="tight")
plt.show()
plt.close()
