import numpy as np
import pandas as pd
import os
from joblib import dump
from typing import Any
from sklearn.metrics import mean_squared_error, r2_score

# Avalia a performance de qualuqer modelo usando RMSE e R2
def evaluate_model_performance(y_true: pd.DataFrame, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calcula um conjunto de métricas de avaliação de regressão para um problema multitarget.

    Args:
        y_true (pd.DataFrame): DataFrame com os valores reais dos alvos.
        y_pred (np.ndarray): Array NumPy com os valores previstos pelo modelo.

    Returns:
        Dict[str, float]: Um dicionário contendo os nomes das métricas e seus respectivos scores.
    """
    # Extrair os nomes das colunas alvo do DataFrame y_true
    target_1_name = y_true.columns[0]
    target_2_name = y_true.columns[1]

    # Calcular as métricas para o primeiro alvo
    r2_target_1 = r2_score(y_true.iloc[:, 0], y_pred[:, 0])
    rmse_target_1 = np.sqrt(mean_squared_error(y_true.iloc[:, 0], y_pred[:, 0]))
    
    # Calcular as métricas para o segundo alvo
    r2_target_2 = r2_score(y_true.iloc[:, 1], y_pred[:, 1])
    rmse_target_2 = np.sqrt(mean_squared_error(y_true.iloc[:, 1], y_pred[:, 1]))
    
    # Montar o dicionário de resultados
    metrics = {
        f"R2 para {target_1_name}": r2_target_1,
        f"R2 para {target_2_name}": r2_target_2,
        f"RMSE para {target_1_name}": rmse_target_1,
        f"RMSE para {target_2_name}": rmse_target_2
    }
    
    # Arredondar os valores para 4 casas decimais para melhor visualização
    metrics = {key: round(value, 4) for key, value in metrics.items()}
    
    return metrics

#-----------------------------------------------------------------------------------------

# Salva os dados de performance do modelo em um csv para facilitar comparação
def save_results(
    model_name: str, 
    metrics: dict[str, float],
    best_params: dict,
    filepath: str 
) -> None:
    """
    Salva as métricas de um modelo em um arquivo CSV, criando o arquivo
    e o cabeçalho se ele não existir, ou anexando uma nova linha se já existir.

    Args:
        model_name (str): Um nome para identificar o modelo (ex: 'Ridge Baseline').
        metrics (Dict[str, float]): O dicionário de métricas retornado pela função de avaliação.
        best_params (Dict): O dicionário de melhores parâmetros retornado pelo GridSearchCV.
        filepath (str): O caminho para o arquivo CSV de resultados.
    """

    # Cria um DataFrame com os novos resultados
    results_df = pd.DataFrame({
        'Model': [model_name],
        'Best_Params': [str(best_params)],
        **metrics
    })
    
    try:
        # Garante que o diretório exista
        output_dir = os.path.dirname(filepath)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(filepath):
            # Se o arquivo não existe, salva com o cabeçalho
            results_df.to_csv(filepath, index=False)
        else:
            # Se o arquivo já existe, anexa sem o cabeçalho
            results_df.to_csv(filepath, mode='a', header=False, index=False)
        
        print(f"Resultados para o modelo '{model_name}' salvos com sucesso em {filepath}")
        
    except (IOError, PermissionError) as e:
        print(f"ERRO: Não foi possível salvar os resultados em '{filepath}'. Erro: {e}")

#-----------------------------------------------------------------------------------------

def save_model(model: Any, filepath: str) -> None:
    """
    Salva um objeto de modelo treinado em um arquivo usando joblib.

    Args:
        model (Any): O objeto do modelo treinado (ex: o pipeline retornado
                    pelo .best_estimator_ do GridSearchCV).
    """

    print(f"Salvando o modelo em: {filepath}...")
    try:
        # Garante que o diretório de destino exista
        output_dir = os.path.dirname(filepath)
        os.makedirs(output_dir, exist_ok=True)
        
        # Salva o modelo no arquivo especificado
        dump(model, filepath)
        
        print("Modelo salvo com sucesso!")
        
    except (IOError, PermissionError) as e:
        print(f"Erro: Não foi possível salvar o modelo em '{filepath}'.")
        raise e
