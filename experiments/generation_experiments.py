import os
import sys
import joblib
import argparse

# Pega o caminho absoluto da pasta onde este script está (/experiments)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Sobe um nível para chegar na raiz do projeto
project_root = os.path.join(current_dir, "..")

# Adiciona a raiz ao buscador do Python
sys.path.append(os.path.abspath(project_root))

from src.pipeline_generator import PipelineGenerator

def run_discovery_pipeline(args):
    # CONFIGURAÇÕES 
    RULES_JSON = "../configs/pipeline_rules.json"
    RANGES_JSON = "../configs/global_hyperparameter_ranges.json"
    SCHEMA_JSON = "../configs/feature_schema.json"

    # CAMINHO PARA CARREGAR OS MODELOS
    if args.model_name == "mlp":
        MODEL_PATH = f"../candidate_models/{args.dataset_name}/{args.model_name}_native.joblib"
    else:
        MODEL_PATH = f"../candidate_models/{args.dataset_name}/{args.model_name}_multi_output.joblib"

    # ONDE SAÍRÃO OS RESULTADOS
    OUTPUT_RANKING = "../experiments_results/inference_ranking"

    # Inicializa o gerador
    gen = PipelineGenerator(RULES_JSON, RANGES_JSON, SCHEMA_JSON)
    
    # Gera as instâncias sintéticas
    df_synthetic = gen.generate_batch(args.dataset_name, args.n_instances)
    path_synthetic = gen.save_for_inference(df_synthetic, args.dataset_name, batch_id=args.batch_id)
    
    # Carrega o modelo
    print(f"\nCarregando modelo {args.model_name.upper()} para avaliar {args.dataset_name.upper()}...")
    model = joblib.load(MODEL_PATH)
    
    # Realiza inferência
    print(f"Fazendo inferencia...")
    predictions = model.predict(df_synthetic)
    
    # Ranqueia e ordena
    df_synthetic['predicted_F1'] = predictions[:, 0]
    df_synthetic['predicted_model_size_log'] = predictions[:, 1]
    df_ranked = df_synthetic.sort_values(by='predicted_F1', ascending=False)
    
    # Garante o diretório e salva o top 30 pipelines
    os.makedirs(OUTPUT_RANKING, exist_ok=True)
    output_filename = f"top_30_{args.dataset_name}_{args.model_name}_batch_{args.batch_id}.csv"
    output_path = os.path.join(OUTPUT_RANKING, output_filename)
    df_ranked.head(30).to_csv(output_path, index=False)
    
    # Printa os resultados
    print(f"Ranqueamento finalizado: {args.dataset_name.upper()}")
    print(f"Melhor F1 Previsto: {df_ranked['predicted_F1'].iloc[0]:.4f}")
    print(f"Top 30 salvo em: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para inferir desempenho de pipelines candidatos")
    
    # Opções de modelo para fazer previsão
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True,
        choices=['xgboost', 'lgbm', 'mlp'], 
        help="Modelo usado para inferir"
    )

    # Opções de pipelines sintéticos
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['birds', 'medical', 'enron', 'scene', 'yeast'], 
        help="Pipelines de qual dataset o script vai gerar"
    )

    # Quantidade de instâncias que o script vai gerar
    parser.add_argument(
        '--n_instances',
        type=int,
        required=True,
        help="Quantidade de instâncias a serem geradas"
    )

    # Id da batch para não sobrescrever nenhum arquivo
    parser.add_argument(
    '--batch_id', 
    type=int, 
    default=1, 
    help="Identificador do lote para evitar sobrescrita de arquivos"
    )

    args = parser.parse_args()
    run_discovery_pipeline(args)