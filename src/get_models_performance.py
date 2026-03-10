import os
import time
import joblib
import pandas as pd
import numpy as np
import warnings

# Ignorar avisos de versões do sklearn (carregar joblibs)
warnings.filterwarnings("ignore")

# Função para medir a performance de um modelo:
# carrega o modelo, mede o tamanho em disco e o tempo médio de inferência.
def measure_performance(archive_path):
    n_features_default = 207
    try:
        # Carregar o modelo
        model = joblib.load(archive_path)
        
        # Pegar o tamanho (em MB)
        size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        
        # Detectar input shape (para criar dummy data)
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        else:
            n_features = n_features_default
            
        # Simula a entrada de um pipeline
        X_dummy = np.random.uniform(low=-1.0, high=1.5, size=(1, n_features))
        
        # Tempo de Inferência (20 execuções)
        # Aquecimento (Warm-up)
        model.predict(X_dummy) 
        
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            model.predict(X_dummy)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            
        time_mean = np.mean(times)
        
        return size_mb, time_mean
        
    except Exception as e:
        return None, None

# Loop Principal 
os.makedirs("./ranking", exist_ok=True)
ready_dfs = []
datasets = ["birds", "enron", "medical", "scene", "yeast"]

print("iniciando Benchmarking dos surrogados...\n")

for dataset in datasets:
    csv_path = f"./results/model_comparison_{dataset}.csv"
    folder = dataset # nome da pasta é igual ao do dataset
    
    print(f"Processando: {dataset}")

    # Leitura e reformatação do CSV
    if not os.path.exists(csv_path):
        print(f"[AVISO] Caminho {csv_path} não encontrado")
        raise FileNotFoundError(f"[ERRO CRÍTICO] CSV não encontrado: {csv_path}")
        
    # Lê o CSV
    df_raw = pd.read_csv(csv_path)
    
    try:
        # Cria um novo DataFrame sem os hiperparâmetros
        df_metrics = df_raw.iloc[:, [0, 2, 3, 4, 5]].copy()
        df_metrics.columns = [
            'Model_Name', 
            'R2_F1', 
            'R2_ModelSize_Log', 
            'RMSE_F1', 
            'RMSE_ModelSize_Log'
        ]
    except IndexError:
        print("[ERRO] O número de colunas não está correto")
        raise ValueError(f"[ERRO CRÍTICO] O dataset {dataset} não tem as 6 colunas esperadas")

    # Coleta da performance
    performance_data = []
    
    for idx, row in df_metrics.iterrows():
        # Extrai o nome do modelo
        model_name = row['Model_Name'] 
        # Monta o caminho completo
        full_path = f"./models/{folder}/{model_name}.joblib"
        if os.path.exists(full_path):
            mb, seconds = measure_performance(full_path)
            
            if mb is not None:
                performance_data.append({
                    'Model_Name': model_name,
                    'Disk_Size_mb': round(mb, 4),
                    'Inference_time': round(seconds, 6)
                })
        else:
            print("[ERRO] caminho completo incorreto")
            raise FileNotFoundError(f"[ERRO CRÍTICO] Modelo faltando: {full_path}")

    performance_df = pd.DataFrame(performance_data)

    # Merge
    if not performance_df.empty:
        # Junta as métricas de qualidade com as de performance 
        df_final = pd.merge(df_metrics, performance_df, on='Model_Name', how='inner')
        df_final['Dataset_Source'] = dataset
        
        # Calcula score para ordenar mais facilmente
        df_final['Weighted_Score'] = (0.65 * df_final['R2_F1']) + (0.35 * df_final['R2_ModelSize_Log'])
        
        # Ordena com base nos pesos definidos acima
        df_final = df_final.sort_values(by='Weighted_Score', ascending=False).reset_index(drop=True)

        # Salva o CSV 
        output_name = f"./ranking/ranked_models_{dataset}.csv"
        df_final.to_csv(output_name, index=False)
        ready_dfs.append(df_final)
        
    else:
        raise RuntimeError(f"[ERRO CRÍTICO] DataFrame de performance vazio para {dataset}.")
    
# --------------------------------------------

# Merge em um dataset mestre
if ready_dfs:
    all_dfs = pd.concat(ready_dfs, ignore_index=True)
    
    # Reordenar colunas para facilitar leitura
    cols_order = [
        'Dataset_Source', 
        'Model_Name', 
        'R2_F1', 
        'R2_ModelSize_Log',     
        'Weighted_Score',
        'Inference_time',    
        'Disk_Size_mb',      
        'RMSE_F1', 
        'RMSE_ModelSize_Log' 
    ]
    
    all_dfs = all_dfs[cols_order]

    # Dentro de cada grupo os melhores modelos estarão primeiro
    all_dfs = all_dfs.sort_values(
        by=['Dataset_Source', 'Weighted_Score'], 
        ascending=[True, False]
    )

    all_dfs.to_csv("./ranking/all_models.csv", index=False)
else:
    raise RuntimeError(f"[ERRO CRÍTICO] lista de dataframes '{ready_dfs}' vazia")