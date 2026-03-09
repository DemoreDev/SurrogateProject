import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.multi_output_training import (train_ridge_regression, 
train_random_forest, train_lgbm, train_catboost, train_xgboost)
from src.evaluation import evaluate_model_performance, save_results, save_model

""" 
Script que permite rodar vários experimentos da abordagem 
"multi_output", variando o dataset de treinamento e o modelo.
"""

def main(args):

    # definindo o caminho do dataset
    data_path = f"../datasets/proc_{args.dataset_name}.csv"
    
    print("Lendo o Dataset...")
    df = pd.read_csv(data_path)
    
    print("\nDividindo dados em treino e teste...")
    X = df.drop(columns=['F1 (macro averaged by label)', 'Model Size', 'Model Size Log'])
    y = df[['F1 (macro averaged by label)', 'Model Size Log']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDivisao concluida! Treinando o modelo: {args.model_name.upper()}")
    
    # Lógica para selecionar o modelo escolhido
    if args.model_name == 'ridge':
        grid_search_results = train_ridge_regression(
            X_train=X_train, 
            y_train=y_train, 
            alpha_values=[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )
        best_model = grid_search_results.best_estimator_
        best_params = grid_search_results.best_params_
    
    elif args.model_name == 'random_forest':
        best_model, best_params, _ = train_random_forest(
            X_train=X_train, 
            y_train=y_train, 
            n_trials=50
        )
    
    elif args.model_name == 'lgbm':
        best_model, best_params, _ = train_lgbm(
            X_train=X_train, 
            y_train=y_train, 
            n_trials=50
        )

    elif args.model_name == 'catboost':
        best_model, best_params, _ = train_catboost(
            X_train=X_train, 
            y_train=y_train, 
            n_trials=50
        )

    elif args.model_name == 'xgboost':
        best_model, best_params, _ = train_xgboost(
            X_train=X_train, 
            y_train=y_train, 
            n_trials=50
        )
        
    else:
        raise ValueError(f"Modelo '{args.model_name}' não reconhecido. Opções: ridge, random_forest, lgbm, catboost, xgboost")

    # Obtendo métricas
    print(f"\nAvaliação do Modelo: {args.model_name.upper()}")
    predictions = best_model.predict(X_test)
    final_metrics = evaluate_model_performance(y_test, predictions)

    # Exibindo métricas
    print("Resultados Finais no Conjunto de Teste:")
    for metric_name, score in final_metrics.items():
        print(f"  - {metric_name}: {score}")

     # Salvar os resultados da performance no CSV de comparação
    print(f"\nSalvando Artefatos: {args.model_name.upper()}")
    results_path = f'../results/model_comparison_{args.dataset_name}.csv'
    save_results(
        model_name=f'{args.model_name}_multi_output',
        metrics=final_metrics,
        best_params=best_params,
        filepath=results_path
    )
    
    # Salvar o objeto do modelo treinado
    model_path = f'../models/{args.dataset_name}/{args.model_name}_multi_output.joblib'
    save_model(
        model=best_model,
        filepath=model_path
    )
    
    print("\nExperimento Concluído com Sucesso!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para rodar experimentos de regressão multitarget.")
    
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True,
        choices=['ridge', 'random_forest', 'lgbm', 'catboost', 'xgboost'], 
        help="O nome do modelo a ser treinado."
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help="Nome do dataset (ex: birds)"
    )
    
    args = parser.parse_args()
    main(args)