import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

Atributos_para_explicacao = 5
USAR_LLM = False
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
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

# Treino: times + atributos + resultado real
treino_df = info_partidas.loc[X_train.index].join(X_train).assign(resultado_real=y_train)

# Teste: times + atributos + resultado real + previsão do modelo + probabilidades
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

# utf-8-sig garante que os acentos abram corretamente no Excel
treino_df.to_csv("dataset_treino.csv", index=False, encoding="utf-8-sig")
teste_df.to_csv("dataset_teste.csv", index=False, encoding="utf-8-sig")
print(f"Arquivos salvos: dataset_treino.csv ({len(treino_df)} linhas) e dataset_teste.csv ({len(teste_df)} linhas)")

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

# Importância global (média do |SHAP| em todas as classes)
importancia_global = np.abs(shap_values.values).mean(axis=(0, 2))
ranking = pd.Series(importancia_global, index=X_test.columns).sort_values(ascending=False)
print("\n10 atributos mais importantes para a previsão dos resultados:")
print(ranking.head(10))

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
    """pos = posição da partida dentro de X_test."""
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

    favor_txt = "; ".join(descrever_atributo(colunas[i], valores[i], sv[i]) for i in a_favor)
    contra_txt = "; ".join(descrever_atributo(colunas[i], valores[i], sv[i]) for i in contra)

    texto = (
        f"A partida entre {mandante} x {visitante}, teve o seguinte resultado previsto pelo modelo \"{CLASSES[pred]}\" "
        f"com {probas[pred]:.0%} de probabilidade"
    )
    texto += " (acertou)." if pred == real else f" (o resultado real foi \"{CLASSES[real]}\")."

    if favor_txt:
        texto += f" Os principais atributos que contribuíram para esta previsão foram: {favor_txt}."
    if contra_txt:
        texto += f" E os que atrapalharam foram: {contra_txt}."

    return texto, pred


def reescrever_com_llm(texto_base):
    """Opcional: pede a um LLM que reescreva o texto sem inventar fatos."""
    from transformers import pipeline

    gerador = pipeline("text-generation", model=LLM_MODEL)
    mensagens = [
        {"role": "system", "content": "Você é um analista esportivo. Responda em português, em no máximo 4 frases."},
        {
            "role": "user",
            "content": (
                "Reescreva a explicação abaixo de forma clara e natural. "
                "Use APENAS as informações dadas, sem inventar estatísticas:\n\n" + texto_base
            ),
        },
    ]
    saida = gerador(mensagens, max_new_tokens=150, do_sample=False)
    return saida[0]["generated_text"][-1]["content"]


# ==========================================================
# 4. EXEMPLO DE USO
# ==========================================================
POSICAO = 1059  # posição da partida em X_test

texto, _ = explicar_partida(POSICAO)
print("\n--- Explicação baseada em SHAP ---")
print(texto)

if USAR_LLM:
    try:
        print("\n--- Versão reescrita pelo LLM ---")
        print(reescrever_com_llm(texto))
    except Exception as e:
        print(f"[Aviso] Não foi possível usar o LLM: {e}")

# Gráfico SHAP da mesma partida (classe prevista)
_, classe_prevista = explicar_partida(POSICAO)
shap.plots.waterfall(shap_values[POSICAO, :, classe_prevista], max_display=10)