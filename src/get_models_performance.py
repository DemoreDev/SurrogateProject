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
        
        # Aquecimento (Warm-up)
        model.predict(X_dummy) 
        
        times = []
        
        # Tempo de Inferência para 70 execuções
        for _ in range(70):
            t0 = time.perf_counter()
            model.predict(X_dummy)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            
        time_mean = np.mean(times)
        
        return size_mb, time_mean
        
    except Exception as e:
        return None, None

# Loop Principal 
os.makedirs("../experiments_results/processed", exist_ok=True)
ready_dfs = []
datasets = ["birds", "enron", "medical", "scene", "yeast"]

print("iniciando Benchmarking dos surrogados...\n")

for dataset in datasets:
    csv_path = f"../experiments_results/raw/raw_{dataset}_results.csv"
    
    # nome da pasta que contém os 
    # modelos é o mesmo do dataset
    folder = dataset 
    
    print(f"Processando: {dataset}")

    # Leitura e reformatação do CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")
        
    # Lê o CSV
    df_raw = pd.read_csv(csv_path)
    
    try:
        # Cria um novo DataFrame sem os hiperparâmetros
        df_metrics = df_raw.iloc[:, [0, 2, 3, 4, 5]].copy()
        df_metrics.columns = [
            'Model Name', 
            'Explicabilidade R2 para F1', 
            'Explicabilidade R2 para Model Size (Log)', 
            'RMSE para F1', 
            'RMSE para Model Size (Log)'
        ]
    except IndexError:
        raise ValueError(f"O dataset {dataset} não tem as colunas esperadas")

    # Cria o vetor para coleta da performance
    performance_data = []
    
    for idx, row in df_metrics.iterrows():
        # Extrai o nome do modelo
        model_name = row['Model Name'] 

        # Monta o path
        full_path = f"../all_models/{folder}/{model_name}.joblib"
        if os.path.exists(full_path):
            # Mede a performance do modelo
            mb, seconds = measure_performance(full_path)
            
            if mb is not None:
                # Adiciona os dados de performance ao vetor
                performance_data.append({
                    'Model Name': model_name,
                    'Disk Size (mb)': round(mb, 4),
                    'Inference time': round(seconds, 6)
                })
        else:
            raise FileNotFoundError(f"Modelo faltando: {full_path}")

    # Transforma os dados de performance em dataframe
    performance_df = pd.DataFrame(performance_data)

    # Merge
    if not performance_df.empty:
        # Junta as métricas de qualidade com as de performance 
        df_final = pd.merge(df_metrics, performance_df, on='Model Name', how='inner')
        df_final['Dataset'] = dataset
        
        # Ordena com base no R2 
        df_final = df_final.sort_values(by='Explicabilidade R2 para F1', ascending=False)

        # Path para salvar os dados processados
        output_name = f"../experiments_results/processed/processed_{dataset}_results.csv"

        # Salva os dados
        df_final.to_csv(output_name, index=False)
        
        # Define o dataframe processado como "pronto"
        ready_dfs.append(df_final)
        
    else:
        raise RuntimeError(f"DataFrame de performance vazio para {dataset}.")

# Merge em um dataset mestre
if ready_dfs:
    all_dfs = pd.concat(ready_dfs, ignore_index=True)
    
    # Reordenar colunas para facilitar leitura
    cols_order = [
        'Dataset', 
        'Model Name', 
        'Explicabilidade R2 para F1', 
        'Explicabilidade R2 para Model Size (Log)',     
        'Inference time',    
        'Disk Size (mb)',      
        'RMSE para F1', 
        'RMSE para Model Size (Log)' 
    ]
    
    all_dfs = all_dfs[cols_order]

    # Dentro de cada grupo os melhores modelos estarão primeiro
    all_dfs = all_dfs.sort_values(
        by=['Dataset', 'Explicabilidade R2 para F1'], 
        ascending=[True, False]
    )

    all_dfs.to_csv("../experiments_results/processed/all_results.csv", index=False)
else:
    raise RuntimeError(f"Lista de dataframes '{ready_dfs}' vazia")