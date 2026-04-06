import pandas as pd
import numpy as np
import json
import random
import os

class PipelineGenerator:
    """ 
    Classe responsável por criar (aleatoriamente) um dataframe com
    N pipelines candidatos. Segue algumas regras definidas nos arquivos
    JSON para não criar nenhum candidato fora dos limites disponíveis
    """
    def __init__(self, rules_path, ranges_path, schema_path):
        # Carrega as probabilidades de uso de cada algoritmo
        with open(rules_path, 'r') as f:
            self.rules = json.load(f)
        
        # Carrega as fronteiras (Min/Max) Globais dos hiperparâmetros
        with open(ranges_path, 'r') as f:
            self.ranges = json.load(f)
            
        # Carrega a ordem exata das colunas
        with open(schema_path, 'r') as f:
            self.columns = json.load(f)
            
        # Mapeamento rápido de Nome -> Índice para performance no preenchimento
        self.feature_to_idx = {col: i for i, col in enumerate(self.columns)}

        # Pré-calcula os grupos (da algoritmos) 
        self.groups = {
            'meka': [c for c in self.columns if c.startswith('meka.') and '-' not in c],
            'weka': [c for c in self.columns if c.startswith('weka.') and '-' not in c],
            'mlfs': [c for c in self.columns if c.startswith('mlfs.') and '-' not in c],
            'sk':   [c for c in self.columns if c.startswith('sklearn.') and '-' not in c]
        }

        # Identifica quais colunas são Flag (algoritmo) e quais são Parâmetro 
        self.flag_index = [i for i, col in enumerate(self.columns) if '-' not in col]
        self.param_index = [i for i, col in enumerate(self.columns) if '-' in col]

    def generate_candidates(self, dataset_name, n_instances):
        """
        Função responsável por criar um dataframe artificial de N instâncias.
        Essa função apenas cria a matriz base e escolhe quais algoritmos 
        serão usados, então chama a função '_initialize_hyperparameters' para preencher 
        os hiperparâmetros dos algoritmos escolhidos. As probabilidades de 
        escolha de algoritmos mudam de acordo com qual dataset será gerado
        """
        print(f"Generating {n_instances} pipelines for {dataset_name.upper()}...")
        
        # Extrai regras gerais e específicas do dataset escolhido
        global_rules = self.rules['pipeline_generation_config']['global_probs']
        ds_rules = self.rules['pipeline_generation_config']['datasets'][dataset_name]
        
        # Cria a matriz base
        data = np.empty((n_instances, len(self.columns)))
        
        # Preenche Flags com 0 e Parâmetros com -1.0
        # Pdrão usado no treinamento dos modelos
        data[:, self.flag_index] = 0
        data[:, self.param_index] = -1.0

        # Preenche a coluna 'feature preprocessing' com 1,
        # pois todos os melhores candidatos usam feature preprocessing
        data[:, self.feature_to_idx['feature preprocessing']] = 1
        
        # Para cada instância 
        for i in range(n_instances):
            # Meka: ativa a Flag do algoritmo escolhido (1) 
            # e preenche seus hiperparâmetros com '_initialize_hyperparameters'
            chosen_meka = random.choice(self.groups['meka'])
            data[i, self.feature_to_idx[chosen_meka]] = 1
            self._initialize_hyperparameters(data[i], chosen_meka)

            # Feature selection: escolhe um dos grupos de algoritmos
            # e escolhe qual algoritmo dentro do grupo vencedor;
            # depois preenche os hiperparâmetros com '_initialize_hyperparameters'
            sel_group = np.random.choice(['mlfs', 'sk'], 
                                         p=[ds_rules['selection_weights']['mlfs'], 
                                            ds_rules['selection_weights']['sklearn']])
            chosen_sel = random.choice(self.groups[sel_group])
            data[i, self.feature_to_idx[chosen_sel]] = 1
            self._initialize_hyperparameters(data[i], chosen_sel)

            # Weka: define se vai ser 1 ou 2 algoritmos (de acordo com o dataframe),
            # então escolhe quais algoritmos e preenche seus hiperparâmetros
            n_weka = np.random.choice([1, 2], 
                                      p=[ds_rules['weka_weights']['1_alg'], 
                                         ds_rules['weka_weights']['2_algs']])
            chosen_wekas = random.sample(self.groups['weka'], k=n_weka)
            for w_alg in chosen_wekas:
                data[i, self.feature_to_idx[w_alg]] = 1
                self._initialize_hyperparameters(data[i], w_alg)

        # Retorna o dataframe pronto para inferência 
        return pd.DataFrame(data, columns=self.columns)

    import random

    def _initialize_hyperparameters(self, row_array, alg_name):
        """
        Preenche os hiperparâmetros baseando-se 
        nos tipos definidos no arquivo JSON
        """
        prefix = f"{alg_name}-"
        params_to_fill = [c for c in self.columns if c.startswith(prefix)]
        
        for param in params_to_fill:
            if param in self.ranges:
                config = self.ranges[param]
                p_min, p_max, p_type = config['min'], config['max'], config['type']
                
                if p_type == 'bool':
                    val = random.choice([0.0, 1.0])
                
                elif p_type == 'int':
                    low = max(10, int(p_min)) if 'n_features' in param else int(p_min)
                    high = max(low, int(p_max))
                    val = random.randint(low, high)
                
                else: # float
                    val = random.uniform(p_min, p_max)
                
                row_array[self.feature_to_idx[param]] = float(val)

    
    def save_for_inference(self, df, dataset_name, batch_id=1):
        """
        Função responsável por salvar o dataframe sintético gerado.
        Salva compactado em .gz porque os dataframes possuem muitas instâncias
        """

        # Define a pasta base
        base_path = "../data/synthetic"
        os.makedirs(base_path, exist_ok=True)
        
        # Define o nome do arquivo e o diretório onde será salvo
        filename = f"synthetic_{dataset_name}_batch_{batch_id}.csv.gz"
        full_path = os.path.join(base_path, filename)
        
        # Salva com compressão
        df.to_csv(full_path, index=False, compression='gzip')
        
        print(f"Dataset completo salvo em: {full_path}")
        return full_path