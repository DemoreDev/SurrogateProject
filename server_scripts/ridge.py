import pandas as pd
import numpy as np
import os
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Alterar estas variáveis para cada experimento
MODEL_NAME = 'ridge_regression_mo' # 'mo' vem de 'multi_output'
DATASET_NAME = 'birds' 

# Construção dos Caminhos ()/Também alterar para outros experimentos
DATA_PATH = f'../../datasets/proc_{DATASET_NAME}.csv'
RESULTS_CSV_PATH = f'../../results/model_comparison_{DATASET_NAME}.csv'
MODEL_OUTPUT_PATH = f'../../models/multi_output/{MODEL_NAME}.joblib'

# Início do Script
print("Início do Treinamento:")
print(f"Modelo: {MODEL_NAME} | Dataset: {DATASET_NAME}")

# Divisão do Dataset
print(f"Carregando dados de: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=['F1 (macro averaged by label)', 'Model Size', 'Model Size Log'])
y = df[['F1 (macro averaged by label)', 'Model Size Log']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dados divididos em treino e teste")

# Otimização e Treinamento do Modelo 
print("Iniciando a otimização de hiperparâmetros com GridSearchCV...")
pipeline = make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge()))
alpha_values = [0.1, 1.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 30.0, 50.0, 75.0, 100.0]
grid = {'multioutputregressor__estimator__alpha': alpha_values}
grid_search = GridSearchCV(estimator=pipeline, param_grid=grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Melhores parâmetros encontrados: {best_params}")
print(f"Melhor R² do Cross Validation: {grid_search.best_score_:.4f}\n")

# Avaliação no Conjunto de Teste 
print("Avaliando o melhor modelo no conjunto de teste...")
predictions = best_model.predict(X_test)
target_1_name = "F1 score"
target_2_name = "Model Size Log"
metrics = {
    f"R2 (explicação) para {target_1_name}": round(r2_score(y_test.iloc[:, 0], predictions[:, 0]), 4),
    f"R2 (explicação) para {target_2_name}": round(r2_score(y_test.iloc[:, 1], predictions[:, 1]), 4),
    f"Erro para {target_1_name}": round(np.sqrt(mean_squared_error(y_test.iloc[:, 0], predictions[:, 0])), 4),
    f"Erro para {target_2_name}": round(np.sqrt(mean_squared_error(y_test.iloc[:, 1], predictions[:, 1])), 4)
}
print(f"Métricas de avaliação: {metrics}")

# Salvamento do Modelo e das Métricas 
# Salvar o objeto do modelo
print(f"Salvando o modelo em: {MODEL_OUTPUT_PATH}...")
try:
    model_dir = os.path.dirname(MODEL_OUTPUT_PATH)
    os.makedirs(model_dir, exist_ok=True)
    dump(best_model, MODEL_OUTPUT_PATH)
    print("Modelo salvo com sucesso!")
except (IOError, PermissionError) as e:
    print(f"ERRO ao salvar o modelo: {e}")
    raise e

# Salvar as métricas no CSV
try:
    results_df = pd.DataFrame({'Model': [MODEL_NAME], 'Best_Params': [str(best_params)], **metrics})
    results_dir = os.path.dirname(RESULTS_CSV_PATH)
    os.makedirs(results_dir, exist_ok=True)
    if not os.path.exists(RESULTS_CSV_PATH):
        results_df.to_csv(RESULTS_CSV_PATH, index=False)
    else:
        results_df.to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)
    print(f"Resultados salvos com sucesso em: {RESULTS_CSV_PATH}")
except Exception as e:
    print(f"ERRO ao salvar os resultados no CSV: {e}")

print("------------------ Processo finalizado! ------------------")