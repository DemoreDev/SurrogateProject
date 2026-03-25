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
        ds_rules = self.rules['datasets'][dataset_name]
        global_rules = self.rules['global_probs']
        
        # Matriz base preenchida com -1.0 
        data = np.full((n_instances, len(self.columns)), -1.0)
        
        # Identifica grupos de colunas (flags)
        meka_algs = self.groups['meka']
        weka_algs = self.groups['weka']
        mlfs_algs = self.groups['mlfs']
        sk_algs   = self.groups['sk']

        for i in range(n_instances):
            # Feature Preprocessing (90% de chance)
            if random.random() < global_rules['feature_preprocessing_prob']:
                data[i, self.feature_to_idx['feature preprocessing']] = 1.0
            
            # MEKA (100% de chance)
            chosen_meka = random.choice(meka_algs)
            data[i, self.feature_to_idx[chosen_meka]] = 1.0
            self._sample_params(data[i], chosen_meka)

            # Selection Group (Sorteia entre MLFS e SKLEARN usando os pesos)
            sel_group = np.random.choice(['mlfs', 'sklearn'], 
                                         p=[ds_rules['selection_weights']['mlfs'], 
                                            ds_rules['selection_weights']['sklearn']])
            
            chosen_sel = random.choice(mlfs_algs if sel_group == 'mlfs' else sk_algs)
            data[i, self.feature_to_idx[chosen_sel]] = 1.0
            self._sample_params(data[i], chosen_sel)

            # WEKA (Sorteia 1 ou 2 algoritmos conforme os pesos)
            n_weka = np.random.choice([1, 2], 
                                      p=[ds_rules['weka_weights']['1_alg'], 
                                         ds_rules['weka_weights']['2_algs']])
            
            chosen_wekas = random.sample(weka_algs, k=n_weka)
            for w_alg in chosen_wekas:
                data[i, self.feature_to_idx[w_alg]] = 1.0
                self._sample_params(data[i], w_alg)

        return pd.DataFrame(data, columns=self.columns)

    # Salva o DataFrame gerado em um CSV comprimido.
    def save_for_inference(self, df, dataset_name, output_dir="../data/synthetic"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Salva o arquivo com numeração sequencial 
        version = 1
        while True:
            file_name = f"synthetic_{dataset_name}_v{version}.csv.gz"
            full_path = os.path.join(output_dir, file_name)
            if not os.path.exists(full_path):
                break
            version += 1

        df.to_csv(full_path, index=False, compression='gzip')
        print(f"Execução {version} salva em: {full_path}")
        return full_path