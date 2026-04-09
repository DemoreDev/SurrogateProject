import pandas as pd
import src.path_config as cfg

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
        e seus hyperparams, retornando as strings/dicts formatados.
        """
        active_algos = self._get_active_algorithms(csv_row)
        fp_algo, meka_algo, weka_algo = self._categorize_algorithms(active_algos)
        fp_params, meka_params, weka_params = self._extract_params(active_algos, csv_row)

        fp_command = self._build_fp_string(fp_algo, fp_params)
        meka_command = self._build_java_string(meka_algo, meka_params)
        weka_command = self._build_java_string(weka_algo, weka_params)

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

        # PRINT DE DEBUG
        print(f"[DEBUG TRANSLATOR] Algoritmos Encontrados -> FP: {fp} | MEKA: {meka} | WEKA: {weka}")
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
                    
                    param_info = self.param_types.get(col, {})
                    
                    param_type = param_info.get("type", "float") # Float como padrão de segurança

                    if param_type == "bool":
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
                        
        # PRINT DE DEBUG
        print(f"[DEBUG TRANSLATOR] Params Extraídos -> FP: {fp_params}")
        print(f"[DEBUG TRANSLATOR] Params Extraídos -> MEKA: {meka_params}")
        print(f"[DEBUG TRANSLATOR] Params Extraídos -> WEKA: {weka_params}")
        return fp_params, meka_params, weka_params


    def _build_java_string(self, algo: str, params: dict):
        """
        Constrói a string do terminal (java); retorna string para
        Meka, e dicionário estruturado para Weka.
        """
        if not algo:
            return ""

        map_categorical = {
            # WEKA
            "weka.classifiers.lazy.KStar": {
                "M": {0: "a", 1: "d", 2: "m", 3: "n"}
            },
            "weka.classifiers.rules.DecisionTable": {
                "E": {0: "acc", 1: "rmse", 2: "mae", 3: "auc"},
                "S": {0: "BestFirst", 1: "GreedyStepwise"}
            },
            "weka.classifiers.bayes.BayesNet": {
                "Q": {
                    0: "weka.classifiers.bayes.net.search.local.TAN",
                    1: "weka.classifiers.bayes.net.search.local.K2 -- -P 1",
                    2: "weka.classifiers.bayes.net.search.local.HillClimber -- -P 1",
                    3: "weka.classifiers.bayes.net.search.local.LAGDHillClimber -- -P 1",
                    4: "weka.classifiers.bayes.net.search.local.TabuSearch -- -P 1"
                }
            },
            # MEKA (Wrapper Mulan)
            "meka.classifiers.multilabel.MULAN.HOMER": {
                "method": {0: "BalancedClustering", 1: "Clustering", 2: "Random"},
                "mll": {0: "BinaryRelevance", 1: "ClassifierChain", 2: "LabelPowerset"}
            },
            # MEKA (Nativo)
            "meka.classifiers.multilabel.BCC": {
                "X": {0: "C", 1: "I", 2: "Ib", 3: "Ibf", 4: "H", 5: "Hbf", 6: "X", 7: "F", 8: "L", 9: "None"}
            },
            "meka.classifiers.multilabel.CT": {
                "X": {0: "C", 1: "I", 2: "Ib", 3: "Ibf", 4: "H", 5: "Hbf", 6: "X", 7: "F", 8: "L", 9: "None"}
                # Se 'payoff_function' for array de string, mapeie o "P" aqui também!
            },
            "meka.classifiers.multilabel.CDT": {
                "X": {0: "C", 1: "I", 2: "Ib", 3: "Ibf", 4: "H", 5: "Hbf", 6: "X", 7: "F", 8: "L", 9: "None"}
            }
        }

        # Tradução dos Índices
        mapped_params = {}
        for flag, val in params.items():
            final_val = val
            
            # Se a flag deste algoritmo estiver no mapa, traduz de int para string
            if algo in map_categorical and flag in map_categorical[algo]:
                try:
                    idx = int(float(val))
                    final_val = map_categorical[algo][flag].get(idx, val)
                    # PRINT DE DEBUG
                    print(f"[DEBUG TRANSLATOR] Mapeou {algo}[{flag}]: {val} -> {final_val}")
                except (ValueError, TypeError):
                    pass # Se falhar, mantém o valor original por segurança
                    
            mapped_params[flag] = final_val

        # Algoritmos Wrapper do MULAN
        if ".MULAN." in algo:
            parts = algo.split(".MULAN.")
            base_mulan = f"{parts[0]}.MULAN"
            mulan_algo = parts[1]
            
            dot_params = []
            # Usamos os parâmetros JÁ TRADUZIDOS!
            for flag, val in mapped_params.items():
                if flag == "normalize":
                    continue
                dot_params.append(str(val))
            
            if dot_params:
                mulan_algo = f"{mulan_algo}.{'.'.join(dot_params)}"
                
            return f"{base_mulan} -S {mulan_algo}"

        # Construção Padrão MEKA / WEKA
        command_parts = [algo]

        for flag, val in mapped_params.items():
            if val is True: 
                command_parts.append(f"-{flag}")
            elif val is False:
                continue
            else: 
                command_parts.append(f"-{flag}")
                command_parts.append(str(val))

        final_string = " ".join(command_parts)

        # Lógica Especial do WEKA (SMO Implícito)
        if "weka.classifiers.functions.supportVector" in algo:
            return {
                "slc": "weka.classifiers.functions.SMO",
                "kernel": final_string
            }
        elif "weka.classifiers" in algo:
            return {
                "slc": final_string,
                "kernel": None
            }

        # Retorna string simples para algoritmos MEKA
        return final_string
    
    
    def _build_fp_string(self, algo: str, params: dict) -> str:
        # Constrói a string para FP (python)
        if not algo:
            return ""

        # Mapeamentos baseados nos arrays de configuração 
        map_method = {
            0: "sklearn.feature_selection.f_classif",
            1: "sklearn.feature_selection.chi2",
            2: "sklearn.feature_selection.mutual_info_classif"
        }
        
        map_estimator = {
            0: "sklearn.ensemble.ExtraTreesClassifier()",
            1: "sklearn.ensemble.RandomForestClassifier()"
        }

        # Constrói a lista no formato 'chave=valor'
        formatted_params = []
        
        for flag, val in params.items():
            final_val = val
            
            # Traduz os índices para os valores reais das strings/objetos
            if flag == "method" and isinstance(val, int):
                final_val = map_method.get(val, val)
                
            elif flag == "estimator" and isinstance(val, int):
                final_val = map_estimator.get(val, val)
                
            elif flag == "neighbors" and isinstance(val, int):
                final_val = 10 if val == 0 else val

            # Adiciona na lista de parâmetros
            formatted_params.append(f"{flag}={final_val}")

        # Junta os parâmetros separando por vírgula e engloba nos parênteses
        params_str = ", ".join(formatted_params)
        
        return f"{algo}({params_str})"