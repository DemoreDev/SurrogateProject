import arff
import pandas as pd
import numpy as np
import tempfile
import os

class DataManager:
    """Gerencia os arquivos ARFF de treino/teste e a criação de arquivos temporários"""

    DATASET_LABELS = {
        'birds': 19,
        'medical': 20,
        'enron': 53,
        'scene': 6,
        'yeast': 14
    }
    
    def __init__(self, base_path, dataset_name, temp_dir):
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.temp_dir = temp_dir
        
        if self.dataset_name not in self.DATASET_LABELS:
            raise KeyError(f"Dataset '{self.dataset_name}' não encontrado no mapeamento de labels.")
            
        self.n_labels = self.DATASET_LABELS[self.dataset_name]

    def get_fold_paths(self, k):
        train_path = os.path.join(self.base_path, f"{self.dataset_name}-train-{k}.arff")
        test_path = os.path.join(self.base_path, f"{self.dataset_name}-test-{k}.arff")
        return train_path, test_path

    def load_arff(self, path):
        with open(path, 'r') as f:
            # encode_nominal=False é vital para manter a paridade de tipos com o Java
            data = arff.load(f, encode_nominal=False, return_type=arff.DENSE)
        
        attrs = np.array(data['attributes'], dtype=object)
        vals = np.array(data['data'], dtype=object)
        
        X = pd.DataFrame(vals[:, :-self.n_labels], columns=[a[0] for a in attrs[:-self.n_labels]])
        y = pd.DataFrame(vals[:, -self.n_labels:], columns=[a[0] for a in attrs[-self.n_labels:]])
        return attrs, X, y

    def save_temp_arff(self, df_x, df_y, attr_metadata, prefix):
        """Salva ARFF temporário com o cabeçalho crítico '@relation ... -C -N'"""
        # MEKA exige aspas simples se o nome tiver caracteres especiais ou espaços
        relation = f"'{self.dataset_name}: -C -{self.n_labels}'"
        
        # Otimização: dicionário para busca rápida de tipos (O(1))
        attr_type_map = {name: dtype for name, dtype in attr_metadata}
        
        attributes = []
        for col in df_x.columns:
            # Busca o tipo original; se não achar, assume NUMERIC
            attr_type = attr_type_map.get(col, 'NUMERIC')
            attributes.append((col, attr_type))
        
        # Labels multi-label são sempre binários no MEKA
        for col in df_y.columns:
            attributes.append((col, ['0', '1']))

        arff_dict = {
            'relation': relation,
            'attributes': attributes,
            'data': pd.concat([df_x, df_y], axis=1).to_numpy(dtype=object)
        }

        # Cria o arquivo e fecha o handle para evitar conflitos de I/O
        temp_f = tempfile.NamedTemporaryFile(dir=self.temp_dir, prefix=prefix, suffix='.arff', delete=False)
        temp_f.close() 

        with open(temp_f.name, 'w', encoding='utf8') as f:
            f.write(arff.dumps(arff_dict))
            
        return temp_f.name