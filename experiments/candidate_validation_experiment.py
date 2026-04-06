import pandas as pd
import argparse
import importlib
from pathlib import Path
from src.meka_wrapper import MekaWrapper
from src.dataset_exporter import DatasetExporter
import sys
import os

# Adiciona a raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configuração de Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
LIB_PATH = BASE_DIR / "meka"

DATASET_LABEL_COUNTS = {
    "birds": 19,
    "medical": 45,
    "enron": 53,
    "scene": 6,   
    "yeast": 14
}

def get_algorithm_details(row):
    """
    Extrai quais algoritmos estão ativos (valor 1.0) 
    e seus respectivos parâmetros na linha do CSV
    """

    mlfs_algo = None
    meka_algo = None
    weka_algo = None
    params = {}

    for col, val in row.items():
        if val == 1.0:
            if col.startswith('mlfs.'): mlfs_algo = col
            elif col.startswith('meka.'): meka_algo = col
            elif col.startswith('weka.'): weka_algo = col
        
        if '-' in col: # parâmetro
            params[col] = val
            
    return mlfs_algo, meka_algo, weka_algo, params

def separate_X_y(path: str, csv_name: str):
    """
    Carrega o CSV e separa em Features (X) e Labels (y)
    baseado no número de labels conhecido para o dataset.
    """

    df = pd.read_csv(path)
    
    # Encontra o número de labels (usa lower() para evitar erros de digitação)
    dataset_key = csv_name.lower()
    if dataset_key not in DATASET_LABEL_COUNTS:
        raise ValueError(f"Erro: O dataset '{csv_name}' não está mapeado em DATASET_LABEL_COUNTS.")
    
    num_labels = DATASET_LABEL_COUNTS[dataset_key]
    
    # No padrão multi-label, os labels costumam ser as ÚLTIMAS colunas
    dfX = df.iloc[:, :-num_labels]
    dfy = df.iloc[:, -num_labels:]
    
    print(f"Dataset {csv_name} carregado: {dfX.shape[1]} features, {dfy.shape[1]} labels.")
    
    return dfX, dfy

def run_validation(csv_name, top_x):
    # Carrega previsões ordenadas
    PREDS_PATH = BASE_DIR / "experiments_results" / "predicted_ranking" / f"best_{csv_name}_xgboost.csv"
    df_preds = pd.read_csv(PREDS_PATH).head(top_x)
    
    # Carrega dados originais
    OG_PATH = BASE_DIR / "data" / "raw" / f"raw_{csv_name}.csv"
    dfX_orig, dfy_orig  = separate_X_y(path=OG_PATH, csv_name=csv_name)
     
    # Inicializa ferramentas
    exporter = DatasetExporter(dataset_name="birds_val")
    wrapper = MekaWrapper(meka_lib_path=str(LIB_PATH))
    
    results = []

    for idx, row in df_preds.iterrows():
        print(f"\n" + "="*50)
        print(f"🚀 INICIANDO CANDIDATO {idx + 1}/{top_x}") 
        print(f"📊 F1 Previsto (XGBoost): {row['predicted_F1']:.4f}")
        
        mlfs_name, meka_name, weka_name, all_params = get_algorithm_details(row)
        print(f"Pipeline: {mlfs_name} -> {meka_name} ({weka_name})")
        
        # Seleção de Atributos
        if mlfs_name is not None:
            try:
                print(f"Executando MLFS ({mlfs_name})...")
                module_path, class_name = mlfs_name.rsplit('.', 1)
                module = importlib.import_module(module_path)
                mlfs_class = getattr(module, class_name)
                
                mlfs_params = {k.split('-')[1]: v for k, v in all_params.items() if k.startswith(mlfs_name)}
                if 'n_features' in mlfs_params: 
                    mlfs_params['n_features'] = int(mlfs_params['n_features'])
                
                selector = mlfs_class(**mlfs_params)
                
                # Execução do Fit e Transform
                dfX_reduced = selector.fit(dfX_orig, dfy_orig).transform(dfX_orig)
                print(f"Seleção concluída. Atributos reduzidos para: {dfX_reduced.shape[1]}")
                
            except Exception as e:
                # Log de erro crítico antes de interromper
                print(f"\n{'!'*60}")
                print(f"ERRO CRÍTICO NO MLFS: {mlfs_name}")
                print(f"O experimento foi interrompido para evitar resultados inconsistentes.")
                print(f"{'!'*60}\n")
                
                # Levanta o erro original para o terminal e para a execução aqui
                raise e 
        else:
            print("Pulando MLFS. Usando todos os atributos.")
            dfX_reduced = dfX_orig
        
        #  Exportação para ARFF
        arff_file = f"temp_val_{idx}.arff"
        exporter.export_to_arff(dfX_reduced, dfy_orig, arff_file)
        print(f"Arquivo temporário criado: {arff_file}")
        
        # Meka  
        print(f"Preparando ambiente meka...")
        # 1. Tratamento para o caso especial MLkNN
        meka_params = []
        weka_params_raw = {}
        
        if "MLkNN" in meka_name:
            # 1. Usamos a classe nativa do MEKA para MLkNN
            actual_meka_classifier = "meka.classifiers.multilabel.IBkk"
            actual_weka_classifier = None 
            
            # 2. Capturamos o valor de K (ex: 55)
            k_val = all_params.get(f"{weka_name}-K", 10)
            weka_params_raw = {'K': int(float(k_val))}
            
            print(f"ℹ️  Usando o MLkNN nativo do MEKA (IBkk) com K={weka_params_raw['K']}")
        else:
            # Lógica normal para BR, CC, etc.
            actual_meka_classifier = meka_name
            actual_weka_classifier = weka_name
            weka_params_raw = {k.split('-')[1]: v for k, v in all_params.items() if k.startswith(weka_name)}

        # 3. Montagem dos parâmetros
        weka_cmd_params = []
        for k, v in weka_params_raw.items():
            weka_cmd_params.append(f"-{k}")
            weka_cmd_params.append(str(v))

        # 3. Execução
        n_labels = DATASET_LABEL_COUNTS[csv_name.lower()]

        real_metrics = wrapper.run_pipeline(
            arff_path=arff_file,
            meka_classifier=actual_meka_classifier,
            num_labels=n_labels,
            weka_classifier=actual_weka_classifier,
            meka_params=meka_params, 
            weka_params=weka_cmd_params
        )
        
        if real_metrics:
            f1_real = real_metrics.get('micro_f1', 0)
            diff = abs(row['predicted_F1'] - f1_real)
            print(f"RESULTADO: F1 Real = {f1_real:.4f} | Diferença: {diff:.4f}")
            
            results.append({
                'mlfs': mlfs_name,
                'meka': meka_name,
                'weka': weka_name,
                'predicted_F1': row['predicted_F1'],
                'f1_real': f1_real
            })
        else:
            print(f"Aviso: MEKA não retornou métricas para este candidato")

        # Limpa arquivo temporário
        if Path(arff_file).exists():
            Path(arff_file).unlink()
            print(f"Arquivo temporário removido.")

    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5, help="Quantidade de pipelines para validar")
    parser.add_argument("--csv", type=str, required=True, help="Nome do CSV que os candidatos serão validados")
    args = parser.parse_args()

    final_report = run_validation(args.csv, args.top)
    print("\n RELATÓRIO FINAL DE VALIDAÇÃO:")
    print(final_report)

    REPORT_PATH = BASE_DIR / "experiments_results" / "validation" / f"validation_{args.csv}.csv"
    final_report.to_csv(REPORT_PATH, index=False)