import os
n_cores = os.cpu_count() // 4
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)
# limita o uso do servidor a 1/4 (no máximo)

import pandas as pd
import optuna
from typing import Tuple, Dict, Any

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
from sklearn.multioutput import RegressorChain 
from sklearn.pipeline import make_pipeline


#--------- Modelos da abordagem RegressorChain ---------
# Utiliza Optuna
def train_rf_regressor(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame,
    n_trials: int = 50  
) -> Tuple[Any, Dict, optuna.study.Study]:
    """
    Cria, otimiza e treina um modelo RandomForest

    Args:
        X_train (pd.DataFrame): DataFrame com as features 
        y_train (pd.DataFrame): DataFrame com os targets 
        n_trials (int): Número de combinações de hiperparâmetros a serem testadas pelo Optuna

    Returns:
        Tuple[Any, Dict, optuna.study.Study]: Uma tupla contendo:
            - O melhor modelo (pipeline) treinado com todos os dados de treino.
            - O dicionário com os melhores hiperparâmetros encontrados.
            - O objeto de estudo completo do Optuna para análises futuras.
    """

    # Esta função interna define um único experimento. O Optuna irá chamá-la 'n_trials' vezes.
    def objective(trial: optuna.Trial) -> float:
        
        # Definir o espaço de busca de hiperparâmetros
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': 42
        }

        # Construir o pipeline com os hiperparâmetros sugeridos
        pipeline = make_pipeline(
            RegressorChain(RandomForestRegressor(**params)) 
        )
        
        # Avaliar o modelo usando validação cruzada
        scores = cross_val_score(
            pipeline,
            X=X_train,
            y=y_train,
            cv=3,
            scoring='r2',
            n_jobs=1 # Roda de forma sequencial
        )
        
        # Retornar o score médio, que o Optuna tentará maximizar
        return scores.mean()

    print(f"Iniciando a otimização com Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    print("\nOtimização concluída.")
    print(f"Melhores parâmetros encontrados: {study.best_params}")
    print(f"Melhor R² (média do cross validation): {study.best_value:.4f}")

    # Optuna encontra os parâmetros, mas não retreina o modelo final
    print("\nRetreinando o modelo com os melhores parâmetros...")
    best_params = study.best_params
    best_pipeline = make_pipeline(
        RegressorChain(RandomForestRegressor(**best_params, random_state=42))
    )
    best_pipeline.fit(X_train, y_train)
    print("Modelo final treinado com sucesso.\n")

    return best_pipeline, study.best_params, study

#------------------------------------------------------------------------------------------------------------

# Utiliza Optuna
def train_lgbm_regressor(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame,
    n_trials: int = 50 
) -> Tuple[Any, Dict, optuna.study.Study]:
    """
    Cria, otimiza e treina um modelo LightGBM

    Args:
        X_train (pd.DataFrame): DataFrame com as features 
        y_train (pd.DataFrame): DataFrame com os targets 
        n_trials (int): Número de combinações a serem testadas pelo Optuna

    Returns:
        Tuple[Any, Dict, optuna.study.Study]: Uma tupla contendo:
            - O melhor modelo (pipeline) treinado.
            - O dicionário com os melhores hiperparâmetros.
            - O objeto de estudo completo do Optuna.
    """
    def objective(trial: optuna.Trial) -> float:
        
        # Definir o espaço de busca de hiperparâmetros 
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state': 42,
            'verbose':1
        }

        # Construir o pipeline com os hiperparâmetros sugeridos
        pipeline = make_pipeline(
            RegressorChain(lgb.LGBMRegressor(**params))
        )
        
        # Avaliar o modelo usando validação cruzada
        scores = cross_val_score(
            pipeline,
            X=X_train,
            y=y_train,
            cv=3,
            scoring='r2',
            n_jobs=1 # Roda de forma sequencial
        )
        
        return scores.mean()

    print(f"Iniciando a otimização com Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    print("\nOtimização concluída.")
    print(f"Melhores parâmetros encontrados: {study.best_params}")
    print(f"Melhor R² (média do cross validation): {study.best_value:.4f}")

    print("\nRetreinando o modelo com os melhores parâmetros...")
    best_params = study.best_params
    best_pipeline = make_pipeline(
        RegressorChain(lgb.LGBMRegressor(**best_params, random_state=42, verbose=1))
    )
    best_pipeline.fit(X_train, y_train)
    print("Modelo final treinado com sucesso.\n")

    return best_pipeline, study.best_params, study

#------------------------------------------------------------------------------------------------------------

# Utiliza optuna
def train_xgb_regressor(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame,
    n_trials: int = 50 
) -> Tuple[Any, Dict, optuna.study.Study]:
    """
    Cria, otimiza e treina um modelo XGBoost

    Args:
        X_train (pd.DataFrame): DataFrame com as features
        y_train (pd.DataFrame): DataFrame com os targets 
        n_trials (int): Número de combinações a serem testadas pelo Optuna

    Returns:
        Tuple[Any, Dict, optuna.study.Study]: Uma tupla contendo:
            - O melhor modelo (pipeline) treinado.
            - O dicionário com os melhores hiperparâmetros.
            - O objeto de estudo completo do Optuna.
    """
    def objective(trial: optuna.Trial) -> float:
        
        # Definir o espaço de busca de hiperparâmetros 
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), 
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'verbosity': 0,  
            'random_state': 42
        }

        pipeline = make_pipeline(
            RegressorChain(xgb.XGBRegressor(**params))
        )
        
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=3,
            scoring='r2',
            n_jobs=1 # Roda de forma sequencial
        )
        
        return scores.mean()

    print(f"Iniciando a otimização com Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    print("\nOtimização concluída.")
    print(f"Melhores parâmetros encontrados: {study.best_params}")
    print(f"Melhor R² (média do cross validation): {study.best_value:.4f}")

    print("\nRetreinando o modelo com os melhores parâmetros...")
    best_params = study.best_params
    best_pipeline = make_pipeline(
        RegressorChain(xgb.XGBRegressor(**best_params, verbosity=0, random_state=42))
    )
    best_pipeline.fit(X_train, y_train)
    print("Modelo final treinado com sucesso.\n")

    return best_pipeline, study.best_params, study