#%%
from src.processamento_de_dados import processamento
from src.treinamento import treinamento
from src.curva_roc import curva_roc

def run_pipeline():

    print(" Carregando e processando dados")
    # Função processamento
    df_analise = processamento()

    print("\nSeparando caracteristicas (X) e churn target (y)...")
    # 3. Separa as caracteristicas (X) e the churn target (y)
    X = df_analise.drop(columns=['churn'])
    y = df_analise['churn']

    print("Treinando modelos ")
    # 4. Call the treinamento function
    df_predict = treinamento(X, y)

   

    print("\nMontando gráfico")
    # 5. Plotando curva ROC
    curva_roc(df_predict)
    
    print("Fim.")


if __name__ == '__main__':
    run_pipeline()
