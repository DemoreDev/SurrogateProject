import sys
from pathlib import Path

# 1. Descobre onde ESTE arquivo (test_real_translator.py) está salvo
current_dir = Path(__file__).resolve().parent

# 2. Sobe um nível para chegar na raiz do projeto (ex: de /experiments para /ProjetosIC)
project_root = current_dir.parent

# 3. Adiciona a pasta 'src' no radar de busca do Python
src_path = project_root / "src"
sys.path.append(str(src_path))

# ==========================================
# Agora os imports vão funcionar perfeitamente!
# ==========================================
import pandas as pd
from src.translator import PipelineTranslator
import src.path_config as cfg

def test_real_csv():
    print("Iniciando Teste com Dados Reais do CSV...\n")
    
    # Defina o caminho exato do seu arquivo CSV contendo os pipelines
    # Substitua "seus_pipelines.csv" pelo nome real do arquivo
    csv_path = cfg.BASE_DIR / "results" / "predicted_pipeline_ranking" / "best_medical_xgboost.csv" 
    
    try:
        # Lê o CSV usando o pandas
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo CSV não encontrado em {csv_path}")
        print("Por favor, atualize a variável 'csv_path' no script com o caminho correto do seu arquivo.")
        return

    print(f"✅ CSV carregado com sucesso! Total de linhas: {len(df)}")
    print("-" * 50)
    
    # Seleciona a primeira linha do dataframe (índice 0)
    # Você pode mudar este número para testar outras linhas depois
    row_idx = 0
    real_row = df.iloc[row_idx].iloc[2:-2]
    
    print(f"Testando a tradução da linha {row_idx}...\n")

    # Instancia o tradutor SEM passar o dicionário mockado.
    # Isso forçará a classe a usar cfg.load_hyperparameters_json()
    translator = PipelineTranslator()

    # Executa a tradução usando a linha real
    fp_cmd, meka_cmd, weka_cmd = translator.translate_row(real_row)

    # Imprime os resultados extraídos
    print("RESULTADOS OBTIDOS:\n")
    print(f"FP Command   : '{fp_cmd}'")
    print(f"MEKA Command : '{meka_cmd}'")
    
    print("WEKA Command (Dict):")
    # Verifica se o weka_cmd foi preenchido corretamente antes de iterar
    if weka_cmd and isinstance(weka_cmd, dict):
        for key, value in weka_cmd.items():
            print(f"  -> {key:<7}: '{value}'")
    else:
        print(f"  -> Nenhum comando WEKA estruturado encontrado: {weka_cmd}")
        
    print("\n" + "-" * 50)

if __name__ == "__main__":
    test_real_csv()