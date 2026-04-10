from pathlib import Path
import sys

BASE_DIR  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import argparse
import csv
import os
import uuid
from src.process_arff import save_arff, read_arff
import src.path_config as cfg
from src.java_executor import MekaExecutor
from src.output_parser import parse_output
from src.csv_translator import PipelineTranslator
import warnings

# Silencia os RuntimeWarnings do Scikit-Learn 
warnings.filterwarnings("ignore", category=RuntimeWarning)

import mlfs.br_skb
import mlfs.br_relieff
import mlfs.d2f_adapted
import mlfs.igmf_adapted
import mlfs.lrfs_adapted
import mlfs.lsmfs_adapted
import mlfs.mdmr_adapted
import mlfs.mlsmfs_adapted
import mlfs.pmu_adapted
import mlfs.ppt_mi_adapted
import mlfs.ppt_relieff
import mlfs.ppt_rfe
import mlfs.ppt_sfm
import mlfs.ppt_skb
import mlfs.scls_adapted

import sklearn.feature_selection

def initialize_output_csv():
    # Cria o arquivo CSV de saída com os cabeçalhos se ele não existir

    if not OUTPUT_CSV.exists():
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True) # Garante que a pasta existe
        
        header = [
            "mlc", "slc", "kernel",               # O Pipeline
            "execution_time",                     # Tempo de execução
            "real_f1", "real_size",               # Métricas Reais
            "predicted_F1",                       # F1 predito
            "predicted_model_size",               # Tamanho predito
            "status", "error"                     # Controle de Sucesso/Falha
        ]

        with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def save_line(line: dict):
    # Adiciona uma única linha ao final do arquivo CSV

    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=line.keys())
        writer.writerow(line)


def main(args):
    print("Iniciando Orquestrador de Experimentos...")

    # Prepara o arquivo de saída
    initialize_output_csv()

    # Inicializa o Executor do Java
    executor = MekaExecutor(memory="8G", timeout_sec=3600)

    # Inicializa o tradutor CSV -> string
    translator = PipelineTranslator()

    # Carrega o CSV de entrada 
    df = pd.read_csv(CSV_PATH)

    # Garante que o valor passado não seja maior que o df
    top_n = min(args.top_n, len(df)) 
    
    # Loop principal (itera pelas linhas do CSV)
    for i in range(top_n):
        print(f"\nValidando pipeline [{i + 1}/{len(df)}]...")

        # Pega a linha atual, o f1 predito e tamanho predito
        actual_line = df.iloc[i, 2:-2] 
        predicted_f1 = df.iloc[i, -2]
        predicted_model_size = df.iloc[i, -1]

        fp_cmd, meka_cmd, weka_cmd = translator.translate_row(actual_line) # traduz o df em linhas de comando
        
        # Insere as variáveis do tradutor no formato que o MekaExecutor exige
        pipeline_info = {
            "mlc": meka_cmd,
            "slc": weka_cmd.get("slc", ""),
            "kernel": weka_cmd.get("kernel", "")
        }

        print(f"[*] Pipeline Traduzido: MLC: {pipeline_info['mlc'].split()[0]} | SLC: {pipeline_info['slc'].split()[0]}")
        
        # Lista para guardar o F1, tamanho e tempo de cada um dos 3 folds
        folds_results = []
        total_time = 0 # Para somar o tempo das 3 execuções

        # Loop secundário (itera sobre os 3 folds)
        for fold in range(3):
            print(f"Executando Fold {fold}...")
            
            # Caminhos dos ARFFs 
            orig_train_path = ARFF_PATH / f"{args.dataset_name.lower()}-train-{fold}.arff"
            orig_test_path  = ARFF_PATH / f"{args.dataset_name.lower()}-test-{fold}.arff"

            # Variáveis que o Java usa
            java_train_path = orig_train_path
            java_test_path = orig_test_path
            
            # Feature Preprocessing
            try:
                if fp_cmd and str(fp_cmd).strip() != "":
                    print("Aplicando Feature Preprocessing...")
                    print(f"Comando de FP: {fp_cmd}")
                    
                    # Lê os arffs 
                    feat_types, dfX_train, dfy_train = read_arff(str(orig_train_path), NUM_LABELS)
                    _, dfX_test, dfy_test = read_arff(str(orig_test_path), NUM_LABELS)
                    
                    # Instancia o algoritmo com eval()
                    fp_algorithm = eval(fp_cmd)

                    # Executa Fit e Transform
                    fp_algorithm.fit(dfX_train, dfy_train)
                    dfX_train_new = fp_algorithm.transform(dfX_train)
                    dfX_test_new = fp_algorithm.transform(dfX_test)

                    # Define caminhos temporários
                    java_train_path = TEMP_DIR / f"temp_train_{uuid.uuid4().hex}.arff"
                    java_test_path = TEMP_DIR / f"temp_test_{uuid.uuid4().hex}.arff"

                    # Salva os novos arquivos temporários no disco
                    save_arff(dfX_train_new, dfy_train, feat_types, NUM_LABELS, args.dataset_name, str(java_train_path))
                    save_arff(dfX_test_new, dfy_test, feat_types, NUM_LABELS, args.dataset_name, str(java_test_path))

                
                # Constrói o Comando 
                cmd, temp_model_path = executor.build_command(
                    translated_pipeline=pipeline_info,
                    train_path=str(java_train_path),
                    test_path=str(java_test_path),
                    num_labels=NUM_LABELS
                )
                
                # Executa o Java
                print("Executando o java...")
                res = executor.execute(cmd, temp_model_path, pipeline_info)
                total_time += res.get("time_sec", 0)

                if res["success"]:
                    text_metrics = parse_output(res["output"])
                    f1 = text_metrics.get("f1_real")
                    size = res.get("model_size")
                    print(f"    [+] Sucesso! F1: {f1} | Tempo: {res.get('time_sec')}s | Tamanho: {size} bytes")
                    
                    # Guarda as métricas dessa rodada específica
                    folds_results.append({
                        "f1": text_metrics.get("f1_real"),
                        "size": res.get("model_size"),
                    })
                else:
                    # PRINT DE DEBUG
                    msg = res.get("error", "Erro Desconhecido").splitlines()[0] if res.get("error") else "Sem output de erro."
                    print(f"    [!] ERRO NO JAVA: {msg}")

            except Exception as e:
                print(f"    [!] ERRO CRÍTICO NO PYTHON (Fold {fold}): {str(e)}")
                print("    [!] O algoritmo falhou matematicamente. Abortando este pipeline...")
                # O 'break' interrompe os outros folds e joga o código para o cálculo final, 
                # que vai registrar esse pipeline como "FALHA" no CSV automaticamente!
                break

            finally:
                if java_train_path != orig_train_path and os.path.exists(java_train_path):
                    os.remove(java_train_path)
                if java_test_path != orig_test_path and os.path.exists(java_test_path):
                    os.remove(java_test_path)
        
        # Salva a mediana dos folds para comparação
        csv_line = {
            "mlc": pipeline_info["mlc"],
            "slc": pipeline_info["slc"],
            "kernel": pipeline_info["kernel"],
            "execution_time": round(total_time, 2),
            "real_f1": None,
            "real_size": None,
            "predicted_F1": predicted_f1,
            "predicted_model_size": predicted_model_size,
            "status": "FALHA",
            "error": ""
        }

        # Filtra apenas os folds que não retornaram None no F1
        folds_validos = [r for r in folds_results if r["f1"] is not None]

        if len(folds_validos) > 0:
            # Ordena a lista de dicionários baseando-se no valor do F1 (do menor pro maior)
            folds_validos.sort(key=lambda x: x["f1"])
            
            # Pega o índice do meio 
            indice_mediana = len(folds_validos) // 2
            fold_mediano = folds_validos[indice_mediana]

            csv_line["real_f1"] = fold_mediano["f1"]
            csv_line["real_size"] = fold_mediano["size"]
            csv_line["status"] = "SUCESSO"
        else:
            csv_line["error"] = "Todos os 3 folds falharam ou não retornaram F1."

        # Salva no disco
        save_line(csv_line)
        print(f"-> Concluído! Status: {csv_line['status']} | F1 (Mediana): {csv_line['real_f1']}")

    print("\nExperimentos finalizados!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para validar o desempenho de pipelines candidatos")

    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['birds', 'medical', 'enron', 'scene', 'yeast'], 
        help="Nome do dataset que os pipelines serão validados (implementados e testados)"
    )

    parser.add_argument(
        '--top_n',
        type=int,
        required=True,
        help="Quantidade de pipelines (linhas) a serem validados"
    )

    args = parser.parse_args()

    dataset_labels = {
    "birds": 19,
    "medical": 45,
    "enron": 53,
    "scene": 6,
    "yeast": 14
    }

    # Configurações de Caminho
    CSV_PATH = BASE_DIR / "results" / "predicted_pipeline_ranking" / f"best_{args.dataset_name.lower()}_xgboost.csv"
    OUTPUT_CSV = BASE_DIR / "results" / "validation" / f"validated_{args.dataset_name.lower()}_pipelines.csv"
    ARFF_PATH = BASE_DIR / "data" / "raw" / f"{args.dataset_name.lower()}"
    TEMP_DIR = BASE_DIR / "temp"
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    NUM_LABELS = dataset_labels[args.dataset_name.lower()]

    main(args)