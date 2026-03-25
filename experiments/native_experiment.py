import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.native_training import train_native
from src.evaluate_model import evaluate_model_performance, save_results, save_model

# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

def main(args):
    """ 
    Esse script é responsável por treinar os modelos 
    da abordagem 'native_adaptation'. Para isso basta passar 
    como argumento o nome do modelo e o nome do conjunto de treinamento
    """

    # Define o caminho do conjunto de treinamento
    DATA_PATH = BASE_DIR / "data" / "meta" / "meta_processed" / f"meta_proc_{args.dataset_name}.csv"
    
    print("Lendo o Dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print("\nDividindo dados em treino e teste...")

    # Define os atributos preditivos
    X = df.drop(columns=['F1 (macro averaged by label)', 'Model Size', 'Model Size Log'])

    # Define os atributos alvo
    y = df[['F1 (macro averaged by label)', 'Model Size Log']]

    # Divide em treino e teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDivisao concluida! Treinando o modelo: {args.model_name.upper()}")
    
    # Realiza o treinamento do modelo escolhido
    best_model, best_params, _ = train_native(
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
        model_name=f'{args.model_name}_native',
        metrics=final_metrics,
        best_params=best_params,
        filepath=results_path
    )
    
    # Salvar o objeto do modelo treinado
    model_path = f'../models/{args.dataset_name}/{args.model_name}_native.joblib'
    save_model(
        model=best_model,
        filepath=model_path
    )
    
    print(f"\nExperimento Concluído com Sucesso! O modelo {args.model_name} foi treinado!")
    print(f"Conjunto usado para o treinamento: {args.dataset_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para rodar experimentos de regressão multitarget.")
    
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True,
        choices=['random_forest', 'knn', 'catboost', 'mlp'], 
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