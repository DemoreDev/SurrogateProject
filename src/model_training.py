import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def train_ridge_regression(X_train: pd.DataFrame, 
                           y_train: pd.DataFrame,
                           alpha_values: list[float] = [0.1, 1.0, 5.0, 10.0, 30.0, 50.0, 75.0, 100.0]
                           ) -> GridSearchCV:
    """
    Cria, otimiza e treina o modelo baseline de Regressão Ridge.

    Args:
        X_train (pd.DataFrame): DataFrame com as features de treinamento.
        y_train (pd.DataFrame): DataFrame com os targets de treinamento.
        alpha_values (List[float], optional): Uma lista de valores alpha para o GridSearchCV testar.
                                              Defaults to [0.1, 1.0, 5.0, ...].

    Returns:
        GridSearchCV: O objeto GridSearchCV treinado.
    """

    # Criar o pipeline:
    # Serve para facilitar a lógica e deixar o código mais simples
    pipeline = make_pipeline(
        StandardScaler(), # escalonar as features
        MultiOutputRegressor(Ridge()) # wrapper + regressor
    )

    # Definir os valores a testar
    grid = {'multioutputregressor__estimator__alpha': alpha_values}

    # Instanciando o grid search 
    grid_search = GridSearchCV(
        estimator=pipeline, # objeto pipeline criado acima
        param_grid=grid, # conjunto de valores a testar
        cv=5, # quantidade de folds
        scoring='r2', # métrica de avaliação
        n_jobs=-1, # Usa todo o poder de processamento
        verbose=1 # Mostra o progresso 
    )

    # Testar cada valor de alpha usando cross validation de 5 folds
    print("Iniciando o GridSearch...")
    grid_search.fit(X_train, y_train)

    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    print(f"Melhor R² (média do cross validation): {grid_search.best_score_:.4f}\n")

    return grid_search

#------------------------------------------------------------------------------------------------------------

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