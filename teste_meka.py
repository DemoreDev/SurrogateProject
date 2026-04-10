import subprocess

def testar_experimento():

    comando = [
        "java",
        "-Xmx8G",
        "-cp", "/home/leodemore/projetoFapesp/ProjetosIC/lib/*",
        "meka.classifiers.multilabel.MULAN",
        
        # --- MUDANÇA AQUI: Inserindo Treino e Teste explicitamente ---
        "-t", "/home/leodemore/projetoFapesp/ProjetosIC/data/raw/medical/medical-train-0.arff", 
        "-T", "/home/leodemore/projetoFapesp/ProjetosIC/data/raw/medical/medical-test-0.arff",  
        # -------------------------------------------------------------
        
        "-C", "45",
        "-d", "/home/leodemore/projetoFapesp/ProjetosIC/temp/temp_model.model",
        "-S", "ECC",
        
        "-verbosity", "3",

        "-W", "weka.classifiers.functions.SMO",
        "--",
        "-K", "weka.classifiers.functions.supportVector.NormalizedPolyKernel -E 4 -L"
    ]

    comando2 = [
        "java",
        "-Xmx8G",
        "-cp", "/home/leodemore/projetoFapesp/ProjetosIC/lib/*",
        
        # --- 1. MLC (O Algoritmo do Meka) ---
        "meka.classifiers.multilabel.BRq",
        
        # --- 2. Treino, Teste e Sistema ---
        "-t", "/home/leodemore/projetoFapesp/ProjetosIC/data/raw/medical/medical-train-0.arff", 
        "-T", "/home/leodemore/projetoFapesp/ProjetosIC/data/raw/medical/medical-test-0.arff",  
        "-C", "45",
        "-d", "/home/leodemore/projetoFapesp/ProjetosIC/temp/temp_model_campeao.model",
        
        # --- 3. Parâmetros do MLC (BRq: P = 10.0) ---
        "-P", "10",
        
        # Verbosity fica aqui, antes de chamar o WEKA
        "-verbosity", "3",

        # --- 4. SLC (O Algoritmo Base do Weka) ---
        "-W", "weka.classifiers.trees.LMT",
        
        # (Opcional, mas muito recomendado pelo Weka para separar parâmetros do classificador base)
        "--", 
        
        # --- 5. Parâmetros do SLC (LMT: M=40, W=2, P=Ativo, A=Ativo) ---
        "-M", "40",
        "-W", "2",
        "-P",
        "-A"
    ]

    print("Iniciando o experimento com Treino e Teste separados...")
    print("-" * 60)
    
    try:
        processo = subprocess.run(
            comando2,
            capture_output=True,
            text=True,
            check=True 
        )

        with open("saida.txt", "w") as f:
            f.write(processo.stdout)
        
        print("✅ EXPERIMENTO CONCLUÍDO COM SUCESSO!\n")
        print("--- RESULTADOS (Métricas geradas no arquivo de Teste) ---")
        
        # Ajustei para imprimir um pouco mais da saída, já que a tabela 
        # de métricas costuma ficar no meio/fim do log
        print(processo.stdout) 


    except subprocess.CalledProcessError as e:
        print("❌ ERRO NA EXECUÇÃO DO JAVA!\n")
        print(f"Código de erro: {e.returncode}")
        print("--- MENSAGEM DE ERRO (STDERR) ---")
        print(e.stderr)

if __name__ == "__main__":
    testar_experimento()