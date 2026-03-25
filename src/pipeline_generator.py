import pandas as pd
import numpy as np
import json
import random
import os

class PipelineGenerator:
    def __init__(self, rules_path, ranges_path, schema_path):
        # Carrega as probabilidades 
        with open(rules_path, 'r') as f:
            self.rules = json.load(f)
        
        # Carrega as fronteiras (Min/Max) Globais
        with open(ranges_path, 'r') as f:
            self.ranges = json.load(f)
            
        # Carrega a ordem exata das colunas
        with open(schema_path, 'r') as f:
            self.columns = json.load(f)
            
        # Mapeamento rápido de Nome -> Índice para performance no preenchimento
        self.feature_to_idx = {col: i for i, col in enumerate(self.columns)}

        # Pré-calcula os grupos no início da classe 
        self.groups = {
            'meka': [c for c in self.columns if c.startswith('meka.') and '-' not in c],
            'weka': [c for c in self.columns if c.startswith('weka.') and '-' not in c],
            'mlfs': [c for c in self.columns if c.startswith('mlfs.') and '-' not in c],
            'sk':   [c for c in self.columns if c.startswith('sklearn.') and '-' not in c]
        }

        # Identifica quem é Flag e quem é Parâmetro 
        self.flag_indices = [i for i, col in enumerate(self.columns) if '-' not in col]
        self.param_indices = [i for i, col in enumerate(self.columns) if '-' in col]

    # Preenche os hiperparâmetros dos algoritmos sorteados usando os ranges globais
    def _sample_params(self, row_array, alg_name):
        # Filtra colunas que são parâmetros do algoritmo (ex: 'meka.alg-P-param')
        prefix = f"{alg_name}-"
        params_to_fill = [c for c in self.columns if c.startswith(prefix)]
        
        for param in params_to_fill:
            if param in self.ranges:
                p_min = self.ranges[param]['min']
                p_max = self.ranges[param]['max']
                # Sorteio uniforme dentro do range extraído dos metadados
                row_array[self.feature_to_idx[param]] = random.uniform(p_min, p_max)

    # Gera N pipelines seguindo as regras de probabilidade do dataset informado
    def generate_batch(self, dataset_name, n_instances=50000):
        print(f"Generating {n_instances} pipelines for {dataset_name.upper()}...")
        
        # Extrai regras específicas
        ds_rules = self.rules['pipeline_generation_config']['datasets'][dataset_name]
        global_rules = self.rules['pipeline_generation_config']['global_probs']
        
        # Cria a matriz base
        data = np.empty((n_instances, len(self.columns)))
        
        # Preenche Flags com 0 e Parâmetros com -1.0
        data[:, self.flag_indices] = 0
        data[:, self.param_indices] = -1.0
        
        for i in range(n_instances):
            # Feature Preprocessing (0 ou 1)
            if random.random() < global_rules['feature_preprocessing_prob']:
                data[i, self.feature_to_idx['feature preprocessing']] = 1
            
            # MEKA (Ativa Flag com 1 e sorteia parametros)
            chosen_meka = random.choice(self.groups['meka'])
            data[i, self.feature_to_idx[chosen_meka]] = 1
            self._sample_params(data[i], chosen_meka)

            # Selection Group (MLFS ou SKLEARN)
            sel_group = np.random.choice(['mlfs', 'sk'], 
                                         p=[ds_rules['selection_weights']['mlfs'], 
                                            ds_rules['selection_weights']['sklearn']])
            chosen_sel = random.choice(self.groups[sel_group])
            data[i, self.feature_to_idx[chosen_sel]] = 1
            self._sample_params(data[i], chosen_sel)

            # WEKA (1 ou 2 algoritmos)
            n_weka = np.random.choice([1, 2], 
                                      p=[ds_rules['weka_weights']['1_alg'], 
                                         ds_rules['weka_weights']['2_algs']])
            chosen_wekas = random.sample(self.groups['weka'], k=n_weka)
            for w_alg in chosen_wekas:
                data[i, self.feature_to_idx[w_alg]] = 1
                self._sample_params(data[i], w_alg)

        return pd.DataFrame(data, columns=self.columns)

    # Salva o DataFrame gerado em um CSV comprimido.
    def save_for_inference(self, df, dataset_name, batch_id=1):
        # Define a pasta base
        base_path = "../data/synthetic"
        os.makedirs(base_path, exist_ok=True)
        
        filename = f"synthetic_{dataset_name}_batch_{batch_id}.csv.gz"
        full_path = os.path.join(base_path, filename)
        
        # Salva com compressão
        df.to_csv(full_path, index=False, compression='gzip')
        
        print(f"   -> Massa bruta salva em: {full_path}")
        return full_path