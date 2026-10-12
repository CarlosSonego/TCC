import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,precision_recall_fscore_support
from sklearn.model_selection import train_test_split

TOP_ATRIBUTOS = 10
CLASSES = {0: "Empate", 1: "Mandante", 2: "Visitante"}
os.makedirs("Graficos_Shap_values_XGB", exist_ok=True)
data_set = pd.read_csv("Campeonato-brasileiro-estatisticas-2015-2023-completo.csv")

def resultado(row):
    if row["Vencedor"] == row["mandante"]:
        return 1
    elif row["Vencedor"] == row["visitante"]:
        return 2
    return 0

data_set["resultado"] = data_set.apply(resultado, axis=1)

remover = [
    "ID", "rodata", "data", "hora", "mandante", "visitante",
    "mandante_Placar", "visitante_Placar", "Vencedor", "resultado",
]

X = data_set.drop(columns=remover)
y = data_set["resultado"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=14, stratify=y
)

model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=326,
    learning_rate=0.05,
    min_child_weight=7,
    gamma=0.8,
    max_depth=8,
    eval_metric="mlogloss",
    random_state=14,
)
model.fit(X_train, y_train)

y_teste = model.predict(X_test)
acuracia = accuracy_score(y_test, y_teste)
print("Acurácia:", acuracia)


explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
colunas = list(X_test.columns)


def salvar(nome):
    caminho = os.path.join("Graficos_Shap_values_XGB", f"{nome}.png")
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Gráfico salvo: {caminho}")


fig, ax = plt.subplots(figsize=(6.5, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_teste, labels=[0, 1, 2],
    display_labels=list(CLASSES.values()),
    cmap="Blues", ax=ax,
)
ax.set_title(f"Matriz de confusão")
ax.set_xlabel("Valor previsto pelo modelo")
ax.set_ylabel("Valor real")
salvar("01_matriz_confusao")

prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_teste, labels=[0, 1, 2], zero_division=0)
largura = 0.25
posicoes = np.arange(3)
fig, ax = plt.subplots(figsize=(8, 5))
for k, (nome, valores) in enumerate([("Precision", prec), ("Recall", rec), ("F1-Score", f1)]):
    barras = ax.bar(posicoes + (k - 1) * largura, valores, largura, label=nome)
    ax.bar_label(barras, fmt="%.2f", fontsize=8)
ax.set_xticks(posicoes)
ax.set_xticklabels(list(CLASSES.values()))
ax.set_ylim(0, 1.1)
ax.set_title("Desempenho por classe")
ax.legend()
salvar("02_metricas_por_classe")

media_abs = np.abs(shap_values.values).mean(axis=0)
total = media_abs.sum(axis=1)
ordem = np.argsort(total)[-TOP_ATRIBUTOS:]

fig, ax = plt.subplots(figsize=(9, 7))
esquerda = np.zeros(len(ordem))
for c in range(3):
    ax.barh(
        [colunas[i] for i in ordem], media_abs[ordem, c],
        left=esquerda, label=CLASSES[c],
    )
    esquerda += media_abs[ordem, c]
ax.set_xlabel("impacto médio na saída do modelo")
ax.set_title(f"Importância dos {TOP_ATRIBUTOS} principais atributos")
ax.legend(title="Classe")
salvar("03_importancia_global_por_classe")

for c, nome_classe in CLASSES.items():
    sv_c = shap_values[:, :, c]
    tag = nome_classe.lower()

    shap.plots.bar(sv_c, max_display=TOP_ATRIBUTOS, show=False)
    plt.title(f"Importância dos atributos — {nome_classe}")
    salvar(f"04_{tag}_barras")

    shap.plots.beeswarm(sv_c, max_display=TOP_ATRIBUTOS, show=False)
    plt.title(f"Beeswarm SHAP — {nome_classe}")
    salvar(f"05_{tag}_beeswarm")

