#%%
import pandas as pd
from sklearn import tree
from sklearn import linear_model
from sklearn import naive_bayes

def treinamento(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:

    arvore = tree.DecisionTreeClassifier()
    logistica = linear_model.LogisticRegression(fit_intercept=True, max_iter=1000)
    naive = naive_bayes.GaussianNB()

    # Ajuste de modelos
    arvore.fit(X, y)
    logistica.fit(X, y)
    naive.fit(X, y)

    # Criando o Dataframe de previsão
    df_predict = pd.DataFrame({'churn': y.values})

    # Previsões
    df_predict['arvore'] = arvore.predict(X)
    df_predict['logistica'] = logistica.predict(X)
    df_predict['naive'] = naive.predict(X)

    # Previsões probabilisticas
    df_predict['arvore_probas'] = arvore.predict_proba(X)[:, 1]
    df_predict['logistica_probas'] = logistica.predict_proba(X)[:, 1]
    df_predict['naive_probas'] = naive.predict_proba(X)[:, 1]

    return df_predict
