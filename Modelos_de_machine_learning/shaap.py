import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

Atributos_para_explicacao = 2
CLASSES = {0: "Empate", 1: "Mandante", 2: "Visitante"}
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

y_pred = model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))

info_partidas = data_set[["ID", "mandante", "visitante", "Vencedor"]]

treino_df = info_partidas.loc[X_train.index].join(X_train).assign(resultado_real=y_train)

probas_teste = model.predict_proba(X_test)
teste_df = (
    info_partidas.loc[X_test.index]
    .join(X_test)
    .assign(
        resultado=y_test,
        resultado_previsto=y_pred,
        previsao_esta_correta=(y_test.values == y_pred),
        prob_de_empate=probas_teste[:, 0],
        prob_do_mandante=probas_teste[:, 1],
        prob_do_visitante=probas_teste[:, 2],
    )
)

treino_df.to_csv("dataset_treino.csv", index=False, encoding="utf-8-sig")
teste_df.to_csv("dataset_teste.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, labels=[0, 1, 2], display_labels=["Empate", "Mandante", "Visitante"],
    cmap="Blues", ax=ax,
)
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.show()

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

importancia_global = np.abs(shap_values.values).mean(axis=(0, 2))
ranking = pd.Series(importancia_global, index=X_test.columns).sort_values(ascending=False)

medias_treino = X_train.mean()

def nome_legivel(col):
    return col.replace("_", " ")

def descrever_atributo(col, valor, shap_val):
    media = medias_treino[col]
    if valor > media:
        nivel = "acima da média"
    elif valor < media:
        nivel = "abaixo da média"
    else:
        nivel = "na média"
    return f"{nome_legivel(col)} ({valor:g}, {nivel} de {media:.1f})"


def explicar_partida(pos):
    idx_original = X_test.index[pos]
    mandante = data_set.loc[idx_original, "mandante"]
    visitante = data_set.loc[idx_original, "visitante"]

    probas = model.predict_proba(X_test.iloc[[pos]])[0]
    pred = int(np.argmax(probas))
    real = int(y_test.iloc[pos])

    sv = shap_values.values[pos, :, pred]
    ordem = np.argsort(sv)[::-1]

    a_favor = [i for i in ordem if sv[i] > 0][:Atributos_para_explicacao]
    contra = [i for i in ordem[::-1] if sv[i] < 0][:Atributos_para_explicacao]

    colunas = X_test.columns
    valores = X_test.iloc[pos].values

    a_favor_txt = "; ".join(descrever_atributo(colunas[i], valores[i], sv[i]) for i in a_favor)
    contra_txt = "; ".join(descrever_atributo(colunas[i], valores[i], sv[i]) for i in contra)

    texto = (
        f"A partida entre {mandante} x {visitante}, teve o seguinte resultado previsto pelo modelo \"{CLASSES[pred]}\" "
        f"com {probas[pred]:.0%} de probabilidade"
    )
    texto += " (acertou).\n" if pred == real else f" (o resultado real foi \"{CLASSES[real]}\").\n"

    if a_favor_txt:
        texto += f" Os principais atributos que contribuíram para esta previsão foram: {a_favor_txt}.\n"
    if contra_txt:
        texto += f" E os que atrapalharam foram: {contra_txt}.\n"

    return texto, pred


POSICAO = 1059

texto, _ = explicar_partida(POSICAO)
print(texto)



