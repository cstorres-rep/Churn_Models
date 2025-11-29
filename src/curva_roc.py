# %%
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

def curva_roc(df_predict: pd.DataFrame):

    # Calculo da curva ROC
    arvore_roc = metrics.roc_curve(df_predict['churn'], df_predict['arvore_probas'])
    logistica_roc = metrics.roc_curve(df_predict['churn'], df_predict['logistica_probas'])
    naive_roc = metrics.roc_curve(df_predict['churn'], df_predict['naive_probas'])

    # Calculo da AUC ROC
    arvore_roc_auc = metrics.roc_auc_score(df_predict['churn'], df_predict['arvore_probas'])
    logistica_roc_auc = metrics.roc_auc_score(df_predict['churn'], df_predict['logistica_probas'])
    naive_roc_auc = metrics.roc_auc_score(df_predict['churn'], df_predict['naive_probas'])

    # Print AUC scores
    print(f"AUC ROC para Arvore de Decisão: {arvore_roc_auc:.2f}")
    print(f"AUC ROC para Regressão Logistica: {logistica_roc_auc:.2f}")
    print(f"AUC ROC para Naive Bayes(Gaussiana) {naive_roc_auc:.2f}")

    # Criando gráfico
    plt.figure(figsize=(10, 7))
    plt.plot(arvore_roc[0], arvore_roc[1], marker='.', label=f'Decision Tree (AUC = {arvore_roc_auc:.2f})')
    plt.plot(logistica_roc[0], logistica_roc[1], marker='.', label=f'Logistic Regression (AUC = {logistica_roc_auc:.2f})')
    plt.plot(naive_roc[0], naive_roc[1], marker='.', label=f'Gaussian Naive Bayes (AUC = {naive_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Classifier')

    plt.title('Curva ROC')
    plt.xlabel('Falso Positivo')
    plt.ylabel('Verdadeiro Positive')
    plt.legend()
    plt.grid(True)
    plt.show()
