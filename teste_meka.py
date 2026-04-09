import subprocess

def testar_experimento():
    comando = [
        "java",
        "-cp", "/home/leodemore/projetoFapesp/ProjetosIC/lib/*",
        "meka.classifiers.multilabel.MULAN",
        
        # --- MUDANÇA AQUI: Inserindo Treino e Teste explicitamente ---
        "-t", "/home/leodemore/projetoFapesp/ProjetosIC/data/raw/medical/medical-train-0.arff", 
        "-T", "/home/leodemore/projetoFapesp/ProjetosIC/data/raw/medical/medical-test-0.arff",  
        # -------------------------------------------------------------
        
        "-C", "45",
        "-S", "ECC",
        "-W", "weka.classifiers.functions.SMO",
        "--",
        "-K", "weka.classifiers.functions.supportVector.NormalizedPolyKernel -E 4 -L"
    ]

    print("Iniciando o experimento com Treino e Teste separados...")
    print("-" * 60)
    
    try:
        processo = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            check=True 
        )
        
        print("✅ EXPERIMENTO CONCLUÍDO COM SUCESSO!\n")
        print("--- RESULTADOS (Métricas geradas no arquivo de Teste) ---")
        
        # Ajustei para imprimir um pouco mais da saída, já que a tabela 
        # de métricas costuma ficar no meio/fim do log
        print(processo.stdout) 
        print("\n[...] (Saída truncada)")

    except subprocess.CalledProcessError as e:
        print("❌ ERRO NA EXECUÇÃO DO JAVA!\n")
        print(f"Código de erro: {e.returncode}")
        print("--- MENSAGEM DE ERRO (STDERR) ---")
        print(e.stderr)

if __name__ == "__main__":
    testar_experimento()