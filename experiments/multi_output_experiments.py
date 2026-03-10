import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.multi_output_training import train_multi_output
from src.evaluate_model import evaluate_model_performance, save_results, save_model

""" 
Script que permite rodar vários experimentos da abordagem 
"multi_output", variando o dataset de treinamento e o modelo.
"""

def main(args):

    # definindo o caminho do dataset
    data_path = f"../data/meta/meta_processed/meta_proc_{args.dataset_name}.csv"
    
    print("Lendo o Dataset...")
    df = pd.read_csv(data_path)
    
    print("\nDividindo dados em treino e teste...")

    # Define os atributos preditivos
    X = df.drop(columns=['F1 (macro averaged by label)', 'Model Size', 'Model Size Log'])

    # Define os atributos alvo
    y = df[['F1 (macro averaged by label)', 'Model Size Log']]

    # Divide em treino e teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDivisao concluida! Treinando o modelo: {args.model_name.upper()}")

    # Realiza o treinamento do modelo escolhido
    best_model, best_params, _ = train_multi_output(
        model_key=args.model_name,
        X_train=X_train, 
        y_train=y_train, 
        n_trials=50
    )

    # Obtém métricas
    print(f"\nAvaliação do Modelo: {args.model_name.upper()}")
    predictions = best_model.predict(X_test)
    final_metrics = evaluate_model_performance(y_test, predictions)

    # Exibe métricas
    print("Resultados Finais no Conjunto de Teste:")
    for metric_name, score in final_metrics.items():
        print(f"  - {metric_name}: {score}")

    # Salva os resultados da performance no CSV para comparação
    print(f"\nSalvando Artefatos: {args.model_name.upper()}")
    results_path = f'../experiments_results/raw/raw_{args.dataset_name}_results.csv'
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
        choices=['ridge', 'random_forest', 'lightgbm', 'catboost', 'xgboost'], 
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