# %%
import pandas as pd


# %%

def processamento():
    df = pd.read_excel('data/dataset_churn.xlsx')

    df['renda anual'] = (df['renda anual'] >= 80000).astype(int)

    df['valor do premio'] = (df['valor do premio'] >= 500).astype(int)

    df['tempo de contrato'] = (df['tempo de contrato'] >= 7 ).astype(int)

    df['sinistros'] =  (df['sinistros'] >= 1).astype(int)

    df['pagamento atrasado'] = (df['pagamento atrasado'] >= 3).astype(int)

    df['nivel de satisfacao'] = (df['nivel de satisfacao']>=4).astype(int)

    df = df.replace({
        'Intermediário': 1,
        'Premium':1,
        'Básico':0
        })

    df = df.replace({
        'Alta': 1,
        'Média':1,
        'Baixa':0
        })

    var = [
        'age', 
        'renda anual', 
        'valor do premio', 
        'tempo de contrato',
        'sinistros', 
        'dependentes',
        'cobertura',
        'frequencia de contrato',
        'pagamento atrasado',
        'nivel de satisfacao']

    df_analise = df[var].copy()
    df_analise['churn'] = df['churn'].copy()

    df_analise

# %%
