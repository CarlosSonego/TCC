import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
os.makedirs("Graficos_RF", exist_ok=True)
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

rforest = RandomForestClassifier(
    n_estimators=100, criterion='entropy', max_depth=11, min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
    oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
    max_samples=None, monotonic_cst=None
)
rforest.fit(X_train, y_train)

y_teste = rforest.predict(X_test)
print(classification_report(y_test, y_teste))

classes = ['mandante', 'empate', 'visitante']

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
plt.savefig(os.path.join("Graficos_RF", "Matriz_confusão_teste.png"), dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Acurácia: {rforest.score(X_test, y_test):.4f}\n")


y_treinamento = rforest.predict(X_train)
print(classification_report(y_train, y_treinamento))

figi, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay.from_predictions(
    y_train,
    y_treinamento,
    labels=classes,
    display_labels=classes,
    cmap='Blues',
    ax=ax,
)
plt.title('Matriz de Confusão do modelo de treinamento')
plt.xlabel('Valor previsto pelo modelo')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig(os.path.join("Graficos_RF", "Matriz_confusão_treinamento.png"), dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"Acurácia: {rforest.score(X_train, y_train):.4f}\n")
