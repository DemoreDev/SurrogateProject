import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

def train_ridge_regression(
    X_train: pd.DataFrame, 
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

def train_random_forest(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame
) -> RandomizedSearchCV:
    """
    Cria, otimiza e treina um modelo RandomForestRegressor 

    Args:
        X_train (pd.DataFrame): DataFrame com as features de treinamento.
        y_train (pd.DataFrame): DataFrame com os targets de treinamento.

    Returns:
        RandomizedSearchCV: O objeto RandomizedSearchCV treinado
    """

    # Criar o pipeline:
    # diferente do modelo ridge, aqui o escalonamento não é necessário
    pipeline = make_pipeline(
        # Novamente wrapper + regressor (dessa vez o regressor é o random forest)
        MultiOutputRegressor(RandomForestRegressor(random_state=42))
    )

    # Definir os valores a testar
    param_distributions = {
        'multioutputregressor__estimator__n_estimators': [500],
        'multioutputregressor__estimator__max_depth': [40],
        'multioutputregressor__estimator__min_samples_split': [5],
        'multioutputregressor__estimator__min_samples_leaf': [1],
        'multioutputregressor__estimator__max_features': ['sqrt'],
        'multioutputregressor__estimator__bootstrap': [False]
    }

    # Instanciando o random search
    random_search = RandomizedSearchCV(
        estimator=pipeline, 
        param_distributions=param_distributions,
        n_iter=1,
        cv=2,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    # Testar os hiperparâmetros usando cross validation de 5 folds
    print("Iniciando o RandomSearch...")
    random_search.fit(X_train, y_train)

    print(f"Melhores parâmetros encontrados: {random_search.best_params_}")
    print(f"Melhor R² (média do cross validation): {random_search.best_score_:.4f}\n")
    
    return random_search