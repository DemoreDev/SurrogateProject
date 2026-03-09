import os
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import RegressorChain 
from sklearn.pipeline import make_pipeline

# Limitação de recursos do servidor
n_cores = os.cpu_count() // 4 # limita a 1/4 do total de processamento
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores) 

# Configuração de Hiperparâmetros 
""" Sugestões de hiperparâmetros que o otimizador 
testará no Random Forest para obter melhores resultados """
def get_rf_regchain_params(trial: optuna.Trial) -> Dict:
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': 42
    }

""" Sugestões de hiperparâmetros que o otimizador 
testará no LightGBM para obter melhores resultados """
def get_lightgbm_params(trial: optuna.Trial) -> Dict:
    return {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 42,
        'verbose': -1 
    }

""" Sugestões de hiperparâmetros que o otimizador 
testará no XGBoost para obter melhores resultados """
def get_xgboost_params(trial: optuna.Trial) -> Dict:
    return {
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

# Mapeamento 
""" Mapeia um identificador (string) para a 
classe do modelo e seus hiperparâmetros """
MODEL_FACTORY = {
    'random_forest': (RandomForestRegressor, get_rf_regchain_params),
    'lightgbm': (lgb.LGBMRegressor, get_lightgbm_params),
    'xgboost': (xgb.XGBRegressor, get_xgboost_params)
}

# Treinamento do modelo escolhido
""" Função principal: treina um modelo 
de acordo com o identificador passado """
def train_regressor_chain(
    model_key: str, # chave identificadora do modelo
    X_train: pd.DataFrame, # Atributos do conjunto de treino
    y_train: pd.DataFrame, # Targets do conjunto de treino
    n_trials: int = 50 # Quantidade de vezes da otimização
) -> Tuple[Any, Dict, optuna.study.Study]:
    
    # Caso o modelo não esteja mapeado
    if model_key not in MODEL_FACTORY:
        raise ValueError(f"Modelo '{model_key}' não incluído. Modelos disponíveis: {list(MODEL_FACTORY.keys())}")
    
    # Descompacta a classe do modelo e os hiperparâmetros
    model_class, param_func = MODEL_FACTORY[model_key]

    def objective(trial: optuna.Trial) -> float:
        params = param_func(trial) # Pega uma sugestão de hiperparâmetros
        pipeline = make_pipeline(RegressorChain(model_class(**params))) # Monta o modelo
        
        # Testa o modelo usando cross-validation de 3 folds
        scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=3, scoring='r2', n_jobs=1
        )
        return scores.mean()

    # Optuna otimiza os hiperparâmetros
    print(f"\nIniciando Otimização: {model_key.upper()}")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Último treino com os melhores parâmetros
    print(f"Melhor R²: {study.best_value:.4f}. Retreinando...")
    best_model = make_pipeline(RegressorChain(model_class(**study.best_params)))
    best_model.fit(X_train, y_train)
    
    # Retorna o modelo, os melhores parâmetros e um objeto com vários dados
    return best_model, study.best_params, study