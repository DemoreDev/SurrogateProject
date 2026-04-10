import subprocess
import re

def extrair_f1(output):
    """Procura a linha do F1 Macro por label e extrai o valor numérico."""
    for linha in output.split('\n'):
        if "F1 (macro averaged by label)" in linha:
            # Captura o número no final da linha (funciona com ponto ou vírgula)
            match = re.search(r"(\d+[.,]\d+)", linha)
            if match:
                return float(match.group(1).replace(',', '.'))
    return None

f1_scores = []
data = "enron"

for i in range(3):
    print(f"\n--- INICIANDO FOLD {i} ---")
    
    comando2 = [
        "java",
        "-Xmx8G",
        "-cp", "/home/leodemore/IC/SurrogateProject/lib/*",
        "meka.classifiers.multilabel.BRq",
        "-t", f"/home/leodemore/IC/SurrogateProject/data/raw/{data}/{data}-train-{i}.arff", 
        "-T", f"/home/leodemore/IC/SurrogateProject/data/raw/{data}/{data}-test-{i}.arff",  
        "-C", "45",
        "-d", f"/home/leodemore/IC/SurrogateProject/temp/temp_model_fold_{i}.model",
        "-P", "10",
        "-verbosity", "3",
        "-W", "weka.classifiers.trees.LMT",
        "--", 
        "-M", "40",
        "-W", "2",
        "-P",
        "-A"
    ]

    try:
        processo = subprocess.run(
            comando2,
            capture_output=True,
            text=True,
            check=True 
        )

        f1_fold = extrair_f1(processo.stdout)
        
        if f1_fold is not None:
            f1_scores.append(f1_fold)
            print(f"✅ Fold {i} concluído. F1: {f1_fold}")
        else:
            print(f"⚠️ Fold {i} concluído, mas o F1 não foi encontrado na saída.")

        # Salva a saída do último fold para conferência se desejar
        with open(f"saida_fold_{i}.txt", "w") as f:
            f.write(processo.stdout)

    except subprocess.CalledProcessError as e:
        print(f"❌ ERRO NO FOLD {i}!")
        print(e.stderr)

# --- CÁLCULO DA MÉDIA FINAL ---
print("\n" + "="*30)
if len(f1_scores) > 0:
    media_f1 = sum(f1_scores) / len(f1_scores)
    print(f"RESULTADO FINAL (Média de {len(f1_scores)} folds)")
    print(f"F1 Macro Médio: {media_f1:.4f}")
else:
    print("Não foi possível calcular a média (nenhum F1 extraído).")
print("="*30)