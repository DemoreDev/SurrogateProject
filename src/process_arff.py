import pandas as pd
import numpy as np
import arff 

def read_arff(path_dataset: str, n_labels: int):
    # Lê um arquivo ARFF e divide em DataFrames do Pandas para as features (X) e labels (y).
    
    with open(path_dataset, 'r', encoding='utf-8') as f:
        arff_frame = arff.load(f, encode_nominal=False, return_type=arff.DENSE)

    attributes_names_types = np.array(arff_frame['attributes'], dtype=object)
    attributes_names = [attr_name for attr_name, attr_type in attributes_names_types]
    
    # Extrai os dados puros
    values = np.array(arff_frame['data'], dtype=object)

    # Faz o split baseado na quantidade de labels informada
    X = values[:, :-n_labels]
    y = values[:, -n_labels:]

    dfX = pd.DataFrame(X, columns=attributes_names[:-n_labels])
    dfy = pd.DataFrame(y, columns=attributes_names[-n_labels:])

    return attributes_names_types, dfX, dfy 


def save_arff(dfX: pd.DataFrame, dfy: pd.DataFrame, feature_names_types: list, n_labels: int, dataset_name: str, output_path: str):
    # Junta os DataFrames X e y e salva como um arquivo .arff pronto para o Meka/Weka.

    relation = f"{dataset_name}: -C -{n_labels}"
    attributes = []
    
    # Monta os atributos X (Features)
    for attr in dfX.columns:
        flag = False
        for name_attr, type_attr in feature_names_types:
            if attr == name_attr:
                attributes.append((name_attr, type_attr)) 
                flag = True
                break

        if not flag:
            attributes.append((attr, 'NUMERIC')) 
            
    # Monta os atributos y (Labels/Targets)
    for col in dfy.columns:
        attributes.append((col, ['0', '1']))

    # PROTEÇÃO 1: Garante que os labels sejam strings para casar com a definição nominal ['0', '1']
    dfy = dfy.astype(str)

    # Junta os dataframes
    df_concat = pd.concat([dfX, dfy], axis=1)
    
    # PROTEÇÃO 2: Substitui np.nan do Pandas pelo None nativo do Python.
    # Assim o liac-arff formata valores nulos corretamente como '?' para o Weka
    df_concat = df_concat.replace({np.nan: None})
    
    # Converte para lista nativa
    values = df_concat.to_numpy(dtype=object).tolist()

    # Monta a estrutura final
    arff_frame = {
        'relation': relation,
        'attributes': attributes,
        'data': values
    }

    arff_save = arff.dumps(arff_frame)

    # Salva direto no caminho
    with open(output_path, 'w', encoding='utf-8') as fp:
        fp.write(arff_save)