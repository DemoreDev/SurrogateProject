import os
import pandas as pd
import optuna
import catboost as cb
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Limitação de recursos do servidor
n_cores = os.cpu_count() // 4 # limita a 1/4 do total de processamento
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)

""" Configuração de Hiperparâmetros:
Sugestões de hiperparâmetros que o otimizador 
testará para obter melhores resultados """
def get_rf_native_params(trial: optuna.Trial) -> Dict:
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': 42
    }

def get_knn_native_params(trial: optuna.Trial) -> Dict:
    return {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'p': trial.suggest_int('p', 1, 2),
        'metric': 'minkowski'
    }

def get_catboost_native_params(trial: optuna.Trial) -> Dict:
    return {
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'depth': trial.suggest_int('depth', 2, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'loss_function': 'MultiRMSE',
        'eval_metric': 'MultiRMSE',
        'allow_writing_files': False,
        'verbose': 0,
        'random_state': 42
    }

def get_mlp_native_params(trial: optuna.Trial) -> Dict:
    # Lógica customizada para arquitetura da rede
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layer_size = trial.suggest_int('layer_size', 32, 128)
    
    return {
        'hidden_layer_sizes': (layer_size,) * n_layers,
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'solver': 'adam',
        'early_stopping': True,
        'max_iter': 500,
        'random_state': 42
    }

""" Mapeamento:
Mapeia um identificador (string) para a classe do modelo, 
seus hiperparâmetros e se usa ou não escalonamento"""
MODEL_FACTORY = {
    'random_forest': (RandomForestRegressor, get_rf_native_params, False),
    'knn': (KNeighborsRegressor, get_knn_native_params, True),
    'catboost': (cb.CatBoostRegressor, get_catboost_native_params, False),
    'mlp': (MLPRegressor, get_mlp_native_params, True)
}

""" Função principal: 
Treina um modelo de acordo 
com o identificador passado """
def train_native(
    model_key: str, # Chave identificadora do modelo
    X_train: pd.DataFrame, # Atributos do conjunto de treino
    y_train: pd.DataFrame, # Targets do conjunto de treino
    n_trials: int = 50 # Quantidade de vezes da otimização
) -> Tuple[Any, Dict, optuna.study.Study]:
    
    # Caso o modelo não esteja mapeado
    if model_key not in MODEL_FACTORY:
        raise ValueError(f"Modelo '{model_key}' não incluído. Modelos disponíveis: {list(MODEL_FACTORY.keys())}")

    # Descompacta o modelo, os hiperparâmetros e o boolean de escalonamento
    model_class, param_func, use_scaler = MODEL_FACTORY[model_key]

    def objective(trial: optuna.Trial) -> float:
        params = param_func(trial) # PEga uma sugestão de parâmetros
        
        # Constrói o pipeline
        steps = [StandardScaler()] if use_scaler else [] # Decide se usa escalonamento ou não 
        steps.append(model_class(**params))              # Apenas o knn usa*
        
        # Monta o modelo
        pipeline = make_pipeline(*steps)
        
        # Testa o modelo usando cross-validation de 3 folds
        scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='r2')
        return scores.mean()

    # Optuna otimiza os hiperparâmetros
    print(f"\nIniciando Otimização: {model_key.upper()}")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Último treino com os melhores parâmetros
    print(f"Melhor R²: {study.best_value:.4f}. Finalizando...")
    
    # Caso especial: Para o MLP, reconstrói os hidden_layer_sizes 
    best_params = study.best_params.copy()
    if model_key == 'mlp':
        n_layers = best_params.pop('n_layers')
        layer_size = best_params.pop('layer_size')
        best_params['hidden_layer_sizes'] = (layer_size,) * n_layers

    final_steps = [StandardScaler()] if use_scaler else []
    final_steps.append(model_class(**best_params))
    
    best_pipeline = make_pipeline(*final_steps)
    best_pipeline.fit(X_train, y_train)
    
    # Retorna o modelo, os melhores parâmetros e um objeto com vários dados
    return best_pipeline, best_params, study