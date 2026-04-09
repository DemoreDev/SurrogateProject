import pandas as pd
import path_config as cfg

class PipelineTranslator:
    """
    Classe responsável por traduzir um CSV em strings 
    formatadas que serão usadas no meka (java)
    """
    
    def __init__(self, param_types_json: dict = None):
        if param_types_json is not None:
            self.param_types = param_types_json # Usado para os testes isolados
        else:
            self.param_types = cfg.load_hyperparameters_json()

    def translate_row(self, csv_row: pd.Series) -> tuple:
        """
        Extrai os algoritmos ativos naquela linha (pipeline)
        e seus hyperparams, retornando uma string/dict para cada algoritmo
        """

        active_algos = self._get_active_algorithms(csv_row)
        fp_algo, meka_algo, weka_algo = self._categorize_algorithms(active_algos)
        fp_params, meka_params, weka_params = self._extract_params(active_algos, csv_row)

        fp_command = self._build_command_string(fp_algo, fp_params)
        meka_command = self._build_command_string(meka_algo, meka_params)
        weka_command = {"slc": None, "kernel": None}
        
        if weka_algo:
            weka_string = self._build_command_string(weka_algo, weka_params)
            
            # Se a classe contiver "Kernel", injetamos o SMO como base
            if "Kernel" in weka_algo:
                weka_command["slc"] = "weka.classifiers.functions.SMO"
                weka_command["kernel"] = weka_string
            # Se não for Kernel (ex: J48, RandomForest), ele vai direto no slc
            else:
                weka_command["slc"] = weka_string

        return fp_command, meka_command, weka_command

    def _get_active_algorithms(self, csv_row: pd.Series) -> list:
        
        algorithm_cols = [col for col in csv_row.index if '-' not in col]

        return [algo for algo in algorithm_cols if csv_row[algo] == 1]

    def _categorize_algorithms(self, active_algos: list) -> tuple:

        fp, meka, weka = None, None, None
        
        for algo in active_algos:
            if "mlfs" in algo or "sklearn" in algo.lower():
                fp = algo
            elif "meka.classifiers" in algo:
                meka = algo
            elif "weka.classifiers" in algo:
                weka = algo
            else:
                print(f"Erro: coluna '{algo}' não pertence à fp, meka ou weka")

        return fp, meka, weka

    def _extract_params(self, algos: list, csv_row: pd.Series) -> tuple:
        # Inicializa dicionários vazios
        fp_params, meka_params, weka_params = {}, {}, {}

        for algo in algos:
            params = {}
            prefix = f"{algo}-"
            param_cols = [col for col in csv_row.index if str(col).startswith(prefix)]

            for col in param_cols:
                val = csv_row[col]
                
                # Ignora parâmetros inativos (-1)
                if val != -1:
                    flag_name = str(col).split('-')[-1]
                    
                    # --- MUDANÇA AQUI: Adaptação para o JSON aninhado ---
                    # Pega o dicionário do parâmetro (ou dict vazio se não achar)
                    param_info = self.param_types.get(col, {})
                    # Pega o tipo, assumindo 'float' como padrão de segurança
                    param_type = param_info.get("type", "float")

                    # Aceita tanto 'bool' quanto 'boolean' para evitar bugs futuros
                    if param_type in ["bool", "boolean"]:
                        if val == 1:
                            params[flag_name] = True
                    else:
                        # Limpa floats que são inteiros redondos
                        if isinstance(val, float) and val.is_integer():
                            params[flag_name] = int(val)
                        else:
                            params[flag_name] = val
            
            # Distribui os parâmetros extraídos para o dicionário correto
            if "mlfs" in algo or "sklearn" in algo.lower():
                fp_params = params
            elif "meka.classifiers" in algo:
                meka_params = params
            elif "weka.classifiers" in algo:
                weka_params = params
                        
        return fp_params, meka_params, weka_params

    def _build_command_string(self, algo: str, params: dict) -> str:
        # Retorna string vazia se o algoritmo não existir na linha
        if not algo:
            return ""

        # Lógica especial para o wrapper do MULAN
        if ".MULAN." in algo:
            parts = algo.split(".MULAN.")
            base_mulan = f"{parts[0]}.MULAN"
            mulan_algo = parts[1]
            command_parts = [base_mulan, "-S", mulan_algo]
        else:
            command_parts = [algo]

        # Injeta os hiperparâmetros formatados
        for flag, val in params.items():
            if val is True: # Flag booleana, ex: -L
                command_parts.append(f"-{flag}")
            else: # Flag com valor numérico, ex: -E 4
                command_parts.append(f"-{flag}")
                command_parts.append(str(val))

        # Junta a lista em uma única string e retorna
        return " ".join(command_parts)