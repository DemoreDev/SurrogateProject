import joblib
import argparse
from src.pipeline_generator import PipelineGenerator
import pandas as pd
from pathlib import Path

# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

def rank_candidates(args):
    """
    Essa função é responsável por usar o gerador de pipelines 
    candidatos 'pipeline_generator.py' e ranquear estes candidatos
    com o modelo surrogado escolhido
    """

    # Define os caminhos (absolutos) para as configurações
    RULES_JSON   = BASE_DIR / "configs" / "pipeline_rules.json"
    RANGES_JSON  = BASE_DIR / "configs" / "global_hyperparameter_ranges.json"
    SCHEMA_JSON  = BASE_DIR / "configs" / "feature_schema.json"

    # Define qual modelo será usado
    if args.model_name == "mlp":
        model_filename = f"{args.model_name}_native.joblib"
    else:
        model_filename = f"{args.model_name}_multi_output.joblib"
    
    # Constrói o caminho dos modelos
    MODEL_PATH = BASE_DIR / "candidate_models" / args.dataset_name / model_filename

    # Caminho para salvar o resultado
    OUTPUT_RANKING = BASE_DIR / "experiments_results" / "predicted_ranking"

    # Caminho para os resultados globais (todas as batchs)
    BEST_PIPELINES_PATH = OUTPUT_RANKING / f"best_{args.dataset_name}_{args.model_name}.csv"

    # Garante que a pasta de saída existe 
    OUTPUT_RANKING.mkdir(parents=True, exist_ok=True)

    # Inicializa o gerador
    gen = PipelineGenerator(RULES_JSON, RANGES_JSON, SCHEMA_JSON)
    
    # Gera os candidatos e salva o dataframe 
    df_synthetic = gen.generate_candidates(args.dataset_name, args.n_instances)
    _ = gen.save_for_inference(df_synthetic, args.dataset_name, batch_id=args.batch_id)
    
    # Carrega o modelo
    print(f"\nCarregando modelo {args.model_name.upper()} para avaliar {args.dataset_name.upper()}...")
    model = joblib.load(MODEL_PATH)
    
    # Realiza inferência
    print(f"Fazendo inferencia de {args.n_instances} candidatos...")
    predictions = model.predict(df_synthetic)
    
    # Ranqueia e ordena
    df_synthetic['predicted_F1'] = predictions[:, 0]
    df_synthetic['predicted_model_size_log'] = predictions[:, 1]
    df_ranked = df_synthetic.sort_values(by='predicted_F1', ascending=False)
    
    # Filtra os 5 melhores 
    current_top = df_ranked.head(5)
    
    if BEST_PIPELINES_PATH.exists():
        df_master = pd.read_csv(BEST_PIPELINES_PATH)
        df_combined = pd.concat([df_master, current_top], ignore_index=True)
    else:
        df_combined = current_top

    # Ordena e limita a 100 candidatos
    df_global_elite = df_combined.sort_values(by='predicted_F1', ascending=False).head(100)

    # Salva o resultado final 
    df_global_elite.to_csv(BEST_PIPELINES_PATH, index=False)

    print(f"🔝 Melhor F1 atual: {df_global_elite['predicted_F1'].iloc[0]:.4f}")

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
    rank_candidates(args)