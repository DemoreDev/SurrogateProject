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
from sklearn.neighbors import KNeighborsRegressor
import catboost as cb
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

"""
A implementação nativa é muito parecida com a implementação com wrapper. 
A única diferença está em remover o wrapper de dentro do pipeline
"""

#--------- Modelos da abordagem NativeAdaptation ---------
# Utiliza Optuna
def train_rf_native(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame,
    n_trials: int = 50  
) -> Tuple[Any, Dict, optuna.study.Study]:
    """
    Cria, otimiza e treina um modelo RandomForest Nativo

    Args:
        X_train (pd.DataFrame): DataFrame com as features 
        y_train (pd.DataFrame): DataFrame com os targets 
        n_trials (int): Número de combinações de hiperparâmetros a serem testados pelo Optuna

    Returns:
        Tuple[Any, Dict, optuna.study.Study]: Uma tupla contendo:
            - O melhor modelo (pipeline) treinado com todos os dados de treino.
            - O dicionário com os melhores hiperparâmetros encontrados.
            - O objeto de estudo completo do Optuna para análises futuras.
    """

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

        # O RandomForestRegressor agora é o único passo no pipeline,
        # sem wrappers de multi_output ou chains
        pipeline = make_pipeline(
            RandomForestRegressor(**params) 
        )
        
        # Avaliar o modelo
        scores = cross_val_score(
            pipeline,
            X=X_train,
            y=y_train,
            cv=3,
            scoring='r2',
            n_jobs=1 
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
        RandomForestRegressor(**best_params, random_state=42)
    )
    best_pipeline.fit(X_train, y_train)
    print("Modelo final treinado com sucesso.\n")

    return best_pipeline, study.best_params, study

#------------------------------------------------------------------------------------------------------------

def train_knn_native(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame,
    n_trials: int = 50  
) -> Tuple[Any, Dict, optuna.study.Study]:
    """
    Cria, otimiza e treina um modelo KNeighborsRegressor Nativo

    Args:
        X_train (pd.DataFrame): DataFrame com as features 
        y_train (pd.DataFrame): DataFrame com os targets 
        n_trials (int): Número de combinações de hiperparâmetros a serem testados pelo Optuna

    Returns:
        Tuple[Any, Dict, optuna.study.Study]: Uma tupla contendo:
            - O melhor modelo (pipeline) treinado com todos os dados de treino.
            - O dicionário com os melhores hiperparâmetros encontrados.
            - O objeto de estudo completo do Optuna para análises futuras.
    """

    def objective(trial: optuna.Trial) -> float:
        
        # Definir o espaço de busca de hiperparâmetros 
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 2) # 1 = Distância Manhattan, 2 = Distância Euclidiana
        }

        # Adicionar o StandardScaler ao pipeline. KNN é muito sensível à escala.
        pipeline = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(**params, metric='minkowski')
            # A métrica "minkowski" é uma fórmula generalizada. Se o 'p' 
            # for 1, se torna Manhattan, mas se for 2 torna-se Euclidiana.
            # Assim, é possível testar diferentes métricas de distância.
        )
        
        scores = cross_val_score(
            pipeline,
            X=X_train,
            y=y_train,
            cv=3,
            scoring='r2',
            n_jobs=1 
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
    
    # Também deve ter o StandardScaler
    best_pipeline = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(**best_params, metric='minkowski')
    )
    best_pipeline.fit(X_train, y_train)
    print("Modelo final treinado com sucesso.\n")

    return best_pipeline, study.best_params, study

#------------------------------------------------------------------------------------------------------------

def train_lgbm_native(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame,
    n_trials: int = 50  
) -> Tuple[Any, Dict, optuna.study.Study]:
    """
    Cria, otimiza e treina um modelo LightGBM Nativo

    Args:
        X_train (pd.DataFrame): DataFrame com as features 
        y_train (pd.DataFrame): DataFrame com os targets 
        n_trials (int): Número de combinações de hiperparâmetros a serem testados pelo Optuna

    Returns:
        Tuple[Any, Dict, optuna.study.Study]: Uma tupla contendo:
            - O melhor modelo (pipeline) treinado com todos os dados de treino.
            - O dicionário com os melhores hiperparâmetros encontrados.
            - O objeto de estudo completo do Optuna para análises futuras.
    """

    def objective(trial: optuna.Trial) -> float:
        
        # Definir o espaço de busca de hiperparâmetros
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True), # L1
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True), # L2
            'random_state': 42,
            'verbose': -1 # Silencia completamente a saida
        }

        # O LGBMRegressor é usado diretamente no pipeline
        pipeline = make_pipeline(
            lgb.LGBMRegressor(**params) 
        )
        
        scores = cross_val_score(
            pipeline,
            X=X_train,
            y=y_train,
            cv=3,
            scoring='r2',
            n_jobs=1
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
        lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
    )
    best_pipeline.fit(X_train, y_train)
    print("Modelo final treinado com sucesso.\n")

    return best_pipeline, study.best_params, study

#------------------------------------------------------------------------------------------------------------

def train_catboost_native(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame,
    n_trials: int = 50 
) -> Tuple[Any, Dict, optuna.study.Study]:
    """
    Cria, otimiza e treina um modelo CatBoost Nativo

    Args:
        X_train (pd.DataFrame): DataFrame com as features 
        y_train (pd.DataFrame): DataFrame com os targets 
        n_trials (int): Número de combinações a serem testados pelo Optuna

    Returns:
        Tuple[Any, Dict, optuna.study.Study]: Uma tupla contendo:
            - O melhor modelo (pipeline) treinado.
            - O dicionário com os melhores hiperparâmetros.
            - O objeto de estudo completo do Optuna.
    """
    def objective(trial: optuna.Trial) -> float:
        
        # Definir o espaço de busca de hiperparâmetros 
        params = {
            'iterations': trial.suggest_int('iterations', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'depth': trial.suggest_int('depth', 2, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True), 
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'allow_writing_files': False,
            'verbose': 0, 
            'random_state': 42
        }

        pipeline = make_pipeline(
            cb.CatBoostRegressor(**params)
        )
        
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=3,
            scoring='r2',
            n_jobs=1 
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
        cb.CatBoostRegressor(**best_params, allow_writing_files=False, verbose=0, random_state=42)
    )
    best_pipeline.fit(X_train, y_train)
    print("Modelo final treinado com sucesso.\n")

    return best_pipeline, study.best_params, study

#------------------------------------------------------------------------------------------------------------

def train_xgb_native(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame,
    n_trials: int = 50 
) -> Tuple[Any, Dict, optuna.study.Study]:
    """
    Cria, otimiza e treina um modelo XGBoost Nativo

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
            xgb.XGBRegressor(**params)
        )
        
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=3,
            scoring='r2',
            n_jobs=1 
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
        xgb.XGBRegressor(**best_params, verbosity=0, random_state=42)
    )
    best_pipeline.fit(X_train, y_train)
    print("Modelo final treinado com sucesso.\n")

    return best_pipeline, study.best_params, study

#------------------------------------------------------------------------------------------------------------

def train_mlp_native(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame,
    n_trials: int = 30  # Reduzido para 30, pois MLPs são mais lentos 
) -> Tuple[Any, Dict, optuna.study.Study]:
    """
    Cria, otimiza e treina um MLPRegressor (Rede Neural) Nativo

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

    def objective(trial: optuna.Trial) -> float:
        
        # Sugere uma arquitetura de rede (1 a 3 camadas, de 32 a 128 neurônios)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layer_size = trial.suggest_int('layer_size', 32, 128)
        hidden_layer_sizes = (layer_size,) * n_layers 

        params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True), 
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'solver': 'adam',
            'early_stopping': True, # Economizar tempo
            'max_iter': 500, 
            'random_state': 42
        }

        # Redes Neurais são bem sensíveis à escala. Por isso, usar StandardScaler
        pipeline = make_pipeline(
            StandardScaler(),
            MLPRegressor(**params) 
        )
        
        scores = cross_val_score(
            pipeline,
            X=X_train,
            y=y_train,
            cv=3,
            scoring='r2',
            n_jobs=1
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
    
    # Recriar a 'hidden_layer_sizes' 
    n_layers = best_params.pop('n_layers')
    layer_size = best_params.pop('layer_size')
    best_params['hidden_layer_sizes'] = (layer_size,) * n_layers
    best_pipeline = make_pipeline(
        StandardScaler(),
        MLPRegressor(**best_params, solver='adam', early_stopping=True, max_iter=500, random_state=42)
    )
    best_pipeline.fit(X_train, y_train)
    print("Modelo final treinado com sucesso.\n")

    return best_pipeline, study.best_params, study