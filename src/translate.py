import pandas as pd
import sklearn.ensemble
import sklearn.tree
import sklearn.linear_model

MAPS = {
    'estimator': [
        'sklearn.ensemble.ExtraTreesClassifier()', 
        'sklearn.ensemble.RandomForestClassifier()'
    ],
    'threshold': [
        "'mean'" 
    ]
}

def extract_params(algo_name, row_data, core_columns, style):
    """Extrai hiperparâmetros e traduz IDs para objetos Python quando necessário"""
    if not algo_name:
        return []
    
    params_list = []
    # Busca colunas que pertencem a este algoritmo e que possuem valor != -1
    param_cols = [c for c in core_columns if c.startswith(f"{algo_name}-") and row_data[c] != -1]
    
    for col in param_cols:
        flag = col.split('-')[-1]
        value = row_data[col]
        
        # Limpa os decimais (.0 -> int)
        if value == int(value): 
            value = int(value)
            
        if style == 'java':
            params_list.append(f"-{flag} {value}")
        else: # Estilo Python
            if flag in MAPS:
                try:
                    translated_val = MAPS[flag][int(value)]
                    params_list.append(f"{flag}={translated_val}")
                except (IndexError, ValueError):
                    params_list.append(f"{flag}={value}")
            else:
                params_list.append(f"{flag}={value}")
                
    return params_list

def translate(row):
    """Traduz uma linha do DataFrame em comandos executáveis para Python e Java."""
    # Assumindo que as colunas de algoritmos/parâmetros estão entre o ID e as métricas
    core_columns = row.index[2:-2]
    
    # Identifica algoritmos ativos (valor 1 na coluna principal)
    active_algos = [col for col in core_columns if '-' not in col and row[col] == 1]
    
    fs_algo = next((a for a in active_algos if 'mlfs' in a.lower()), None)
    sklearn_algo = next((a for a in active_algos if 'sklearn' in a.lower()), None)
    meka_algo = next((a for a in active_algos if 'meka' in a.lower()), None)
    weka_algo = next((a for a in active_algos if 'weka' in a.lower()), None)

    # Prioridade para algoritmos de Feature Selection (Python)
    selected_fp = fs_algo if fs_algo else sklearn_algo

    # Montagem do comando de Feature Selection (Python)
    fp_command = None
    if selected_fp:
        fp_params = ", ".join(extract_params(selected_fp, row, core_columns, style='python'))
        fp_command = f"{selected_fp}({fp_params})"
    
    # Montagem do comando MEKA (Java)
    meka_command = None
    if meka_algo:
        meka_params = " ".join(extract_params(meka_algo, row, core_columns, style='java'))
        meka_command = f"{meka_algo} {meka_params}"
    
    # Montagem do comando WEKA (Java)
    weka_command = None
    if weka_algo:
        weka_params = " ".join(extract_params(weka_algo, row, core_columns, style='java'))
        weka_command = f"{weka_algo} {weka_params}"

    return {
        'fp_command': fp_command,
        'meka_command': meka_command,
        'weka_command': weka_command,
        'predicted_f1': row.iloc[-2],
        'predicted_size': row.iloc[-1]
    }