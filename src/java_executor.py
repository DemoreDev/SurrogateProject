import subprocess
import time
import uuid 
import os
import json
from pathlib import Path
import src.path_config as cfg

# Define a raiz do projeto 
BASE_DIR = Path(__file__).resolve().parent.parent

# Define o caminho da pasta temp
TEMP_DIR = BASE_DIR / "temp"

# Garante que a pasta existe 
TEMP_DIR.mkdir(parents=True, exist_ok=True)

class MekaExecutor:
    # Responsável por formatar e executar comandos do Meka/Weka via terminal (Java).
    
    def __init__(self, lib_path: Path = None, memory: str = "8G", timeout_sec: int = 3600):
        # Caso libpath seja passado
        if lib_path is not None:
            self.classpath = f"{lib_path}/*"
        else:
            self.classpath = cfg.MEKA_CLASSPATH
        
        self.memory = f"-Xmx{memory}"
        self.timeout = timeout_sec

    def build_command(self, translated_pipeline: dict, train_path: str, test_path: str, num_labels: int) -> tuple:
        # Transforma o pipeline traduzido na lista exata que o subprocess do Python exige.
        
        mlc_string = translated_pipeline.get("mlc", "")
        if not mlc_string:
            raise ValueError("O algoritmo MLC (Meka) não pode ser vazio.")

        # Base do Comando Java
        cmd = [
            "/home/lddemore@posgrad.usricmc.icmc.usp.br/anaconda3/envs/minhaic/bin/java",
            self.memory,          
            "-cp", self.classpath 
        ]

        # Processa o MLC 
        mlc_parts = mlc_string.split(" ")
        main_class = mlc_parts[0]  # O nome da classe base Meka
        rest_mlc = mlc_parts[1:]   # Os parâmetros Meka (se existirem)

        cmd.append(main_class) # Adiciona classe principal (ex: meka.classifiers.multilabel.MULAN)

        temp_model_path = TEMP_DIR / f"temp_model_{uuid.uuid4().hex}.model"
        
        # Adiciona as obrigações do DataSet logo após a classe principal
        cmd.extend([
            "-t", str(train_path),
            "-T", str(test_path),
            "-C", str(num_labels),
            "-d", str(temp_model_path)
        ])
        
        # Adiciona o restante dos parâmetros do Meka (ex: '-S', 'ECC')
        cmd.extend(rest_mlc)

        cmd.extend([
            "-verbosity", "3"
        ])

        # Adiciona o SLC (Weka Base)
        slc_string = translated_pipeline.get("slc")
        if slc_string:
            cmd.append("-W")
            # Divide os parâmetros. Ex: '...RandomForest -I 10' -> ['...RandomForest', '-I', '10']
            cmd.extend(slc_string.split())

        # Adiciona o Kernel (se houver)
        kernel_string = translated_pipeline.get("kernel")
        if kernel_string:
            cmd.extend(["--", "-K"])
            cmd.append(kernel_string)

        # PRINT DE DEBUG
        print(f"\n[DEBUG EXECUTOR] Comando montado:\n{' '.join(cmd)}\n")
        return cmd, str(temp_model_path)
    

    def execute(self, command_list: list, temp_model_path: str, pipeline_info: dict = None) -> dict:
        # Executa o comando montado e captura a saída.
        
        start_time = time.time()
        pipeline_info = pipeline_info or {} 
        model_size_bytes = None
        
        print("[DEBUG EXECUTOR] Iniciando subprocess do java")

        try:
            result = subprocess.run(
                command_list,
                capture_output=True, 
                text=True,           
                timeout=self.timeout
            )
            
            print("[DEBUG EXECUTOR] subprocess do java finalizado!")
            elapsed_time = time.time() - start_time
            success = (result.returncode == 0)

            if success and os.path.exists(temp_model_path):
                # Pega o tamanho do arquivo em bytes 
                model_size_bytes = os.path.getsize(temp_model_path)
                # Deleta o modelo imediatamente
                os.remove(temp_model_path)
            
            # Pega o erro real. Se o stderr estiver vazio, tenta achar pistas no stdout
            error_desc = result.stderr.strip()
            if not success and not error_desc:
                error_desc = f"Erro no stdout ou processo morto (código {result.returncode}):\n{result.stdout.strip()}"

            response = {
                "success": success,
                "time_sec": round(elapsed_time, 2),
                "output": result.stdout,
                "error": error_desc if not success else "",
                "returncode": result.returncode,
                "model_size": model_size_bytes
            }
            
            if not success:
                self._log_failure(command_list, pipeline_info, response)
                
            return response
        
        # Se der timeout, salva os dados
        except subprocess.TimeoutExpired:
            print(f"      [!] TIMEOUT: O modelo demorou mais de {self.timeout}s e foi abortado.")
            elapsed_time = time.time() - start_time
            response = {
                "success": False,
                "time_sec": round(elapsed_time, 2),
                "output": "",
                "error": f"TIMEOUT: O processo excedeu {self.timeout} segundos.",
                "returncode": None
            }
            self._log_failure(command_list, pipeline_info, response)
            # Apaga um modelo temporário (caso tenha sido criado)
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            
            return response
            
        # Se deu erro por outro motivo
        except Exception as e:
            print(f"      [!] ERRO CRÍTICO NO PYTHON: {str(e)}")
            response = {
                "success": False,
                "time_sec": round(time.time() - start_time, 2),
                "output": "",
                "error": f"ERRO INTERNO: {str(e)}",
                "returncode": None
            }
            self._log_failure(command_list, pipeline_info, response)
            # Apaga um modelo temporário (caso tenha sido criado)
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            return response
        
        
    def _log_failure(self, command_list: list, pipeline_info: dict, execution_result: dict):
        """
        Salva o contexto completo do erro em um arquivo JSON para análise futura.
        """
        log_entry = {
            "command": " ".join(command_list), # Facilita copiar e colar no terminal para testar
            "pipeline_context": pipeline_info, # Aqui ficam as flags, hyperparams, etc.
            "execution_details": execution_result
        }
        
        # Salva em um arquivo com append (cada linha sendo um JSON válido ajuda na leitura posterior)
        with open("error_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")