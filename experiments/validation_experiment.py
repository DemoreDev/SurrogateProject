import pandas as pd
import numpy as np
import argparse
import os
from src.translate import translate
from src.meka_adapted4 import MekaAdapted
from src.data_manager import DataManager
import sklearn.ensemble
from pathlib import Path
import mlfs
from mlfs import (br_relieff, br_skb, d2f_adapted, igmf_adapted, lrfs_adapted,
                  lsmfs_adapted, mdmr_adapted, mlsmfs_adapted, pmu_adapted, ppt_mi_adapted,
                  ppt_relieff, ppt_rfe, ppt_sfm, ppt_skb, scls_adapted)

# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

def run_full_validation(df_predictions, data_manager, java_cmd, meka_cp, top_n: int):
    """Executa a validação real para os top pipelines do ranking"""

    top_n = min(top_n, len(df_predictions))
    
    # Avalia os melhores N resultados
    for i in range(top_n):
        row = df_predictions.iloc[i]
        pipe = translate(row) # Retorna um dicionário com os dados formatados
        
        print(f"\n>>> PIPELINE #{i+1} | PREDICTED F1: {pipe['predicted_f1']:.4f}")
        
        fold_f1_scores = []
        
        for k in range(3): # Folds 0, 1 e 2
            train_path, test_path = data_manager.get_fold_paths(k)
            
            # Carrega dados do fold (meta armazena a estrutura dos dados)
            meta, X_train, y_train = data_manager.load_arff(train_path)
            _, X_test, y_test = data_manager.load_arff(test_path)
            
            # Feature Selection (Python)
            selector = eval(pipe['fp_command'])
            X_train_red = selector.fit_transform(X_train, y_train)
            X_test_red = selector.transform(X_test)
            
            # Salvar ARFFs reduzidos
            tmp_train = data_manager.save_temp_arff(X_train_red, y_train, meta, f"train_f{k}_")
            tmp_test = data_manager.save_temp_arff(X_test_red, y_test, meta, f"test_f{k}_")
            
            # Executar MEKA/WEKA (Java)
            meka = MekaAdapted(
                meka_classifier=pipe['meka_command'],
                weka_classifier=pipe['weka_command'],
                java_command=java_cmd,
                meka_classpath=meka_cp,
                timeout=600
            )
            
            try:
                # O fit_predict retorna o dicionário de estatísticas capturado do terminal
                meka.fit_predict(len(X_test_red), data_manager.n_labels, tmp_train, tmp_test)
                f1_macro = meka.statistics.get('F1 (macro averaged by label)', 0)
                fold_f1_scores.append(f1_macro)
            finally:
                # Limpeza de arquivos temporários
                os.unlink(tmp_train)
                os.unlink(tmp_test)

        # Média final do pipeline nos 3 folds
        avg_f1 = np.mean(fold_f1_scores)
        error = abs(avg_f1 - pipe['predicted_f1'])
        
        print(f"    REAL F1 (AVG): {avg_f1:.4f} | PREDICTED: {pipe['predicted_f1']:.4f}")
        print(f"    ABSOLUTE ERROR: {error:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para validar o desempenho real de pipelines preditos")
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['birds', 'medical', 'enron', 'scene', 'yeast'], 
        help="Qual pipeline o script irá validar"
    )

    parser.add_argument(
        '--top_n',
        type=int,
        required=True,
        help="Quantidade de pipelines a serem validados"
    )

    args = parser.parse_args()
    
    dataset_name = (args.dataset_name).lower()
    BASE_DATA_PATH = str(BASE_DIR / "data" / "raw" / f"{dataset_name}")
    JAVA_EXEC = str("/home/leodemore/anaconda3/envs/ic/bin/java")
    MEKA_LIBS = str(BASE_DIR / "lib" / "*")
    TEMP_DIR = BASE_DIR / "temp"
    PREDS_PATH = str(BASE_DIR / "results" / "predicted_pipeline_ranking" / f"best_{dataset_name}_xgboost.csv")

    TEMP_DIR.mkdir(parents=True, exist_ok=True) # Cria a pasta se ela não existir
    
    datamanager = DataManager(BASE_DATA_PATH, dataset_name, str(TEMP_DIR))
    df_preds = pd.read_csv(PREDS_PATH)
    
    run_full_validation(df_preds, datamanager, JAVA_EXEC, MEKA_LIBS, args.top_n)