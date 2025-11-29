# Projeto Churn Icatu

Neste projeto, usamos metricas comuns de segurados para tentar prever seu momento de churn

## Visão Geral

O objetivo principal é construir modelos de classificação para identificar  a possibilidade de Churn de segurados. Usamos 3 modelos probabilisticos para fazer o ajustes dos dados: Regressão Logistica, Arvore de Decisões e Naive Bayes. Por fim, comparamos auto avaliação de um segurado para permanência ou não no produto com a previsão probabilistica de nossos modelos.
Desta forma, identificamos Verdadeiros Positivos, Verdadeiros Negativos, Falso Positivo, Falso Negativo

Verdadeiros Positvos: Quando um cliente diz ter vontade de permanecer e nosso modelo prediz isso
Verdadeiros Negativos: Quando um cliente diz ter vontade de sair e nosso modelo prediz isso
Falso Positvo: Quando um cliente diz ter vontade de sair e nosso modelo prediz o contrário
Falso Negativo: Quando um cliente diz ter vontade de ficar e nosso modelo prediz o contrário

## Estrutura do Projeto

- `data/`: Contém os dados brutos e processados.
- `notebooks/`: Notebooks Colab utilizados para análise exploratória de dados (EDA) e prototipagem de modelos.
- `src/`: Código-fonte principal organizado em módulos Python para pré-processamento, treinamento de modelos e avaliação.
- `requirements.txt`: Dependências do Python necessárias.

## Ampliação

Futuramente novos modelos probabilisticos devem ser implementados, assim como ferramentas como MLflow e Streamlit 
