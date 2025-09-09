import pandas as pd
import numpy as np
import os

def process_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Carrega um dataset bruto, aplica todas as etapas de limpeza e processamento,
    salva o DataFrame limpo em um novo arquivo .csv e retorna o DataFrame processado

    Args:
        input_path (str): O caminho para o arquivo .csv bruto
        output_path (str): O caminho onde o arquivo .csv processado será salvo
    """
    print("Iniciando o processamento do meta-dataset...")

    # Carregar os dados
    try:
        df = pd.read_csv(input_path, sep=';')
        print(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas")
        original_rows = len(df) # Salvando número de linhas original
    except FileNotFoundError as e:
        print(f"ERRO: O arquivo de entrada não foi encontrado em '{input_path}'.")
        raise e
    
    print("\nRemovendo linhas NaN...")

    # Remover execuções onde os targets não tem valor (NaNs)
    df.dropna(subset=['F1 (macro averaged by label)', 'Model Size'], inplace=True)
    print(f"Removidas {original_rows - len(df)} linhas. Restaram {len(df)} linhas validas")

    print("\nProcessando todas as colunas...")

    # processando e binarizando a coluna "feature preprocessing"
    df = df.rename(columns={'NO_FEATURE_PREPROCESSING': 'feature preprocessing'})
    df['feature preprocessing'] = df['feature preprocessing'].apply(lambda x: 0 if x == 0 else 1)

    # Processando e binarizando as colunas categóricas (nomes dos algoritmos)
    columns = [column for column in df.columns[2:-2] if '-' not in column]
    for column in columns:
        df[column] = df[column].apply(lambda x: 1 if x == 0 else 0)

    # Processando e binarizando as colunas numéricas (hiperparâmetros)
    columns_to_fill = [
        col for col in df.columns[2:-2]
        if col not in columns
    ]
    df[columns_to_fill] = df[columns_to_fill].fillna(-1)

    # Verificações finais
    print("\nProcessamento concluido!:")
    clean_df = df.copy() # Evita warning de fragmentação

    # Criando a coluna "model size log"
    clean_df['Model Size Log'] = np.log1p(clean_df['Model Size'])
    nans = clean_df.isnull().sum().sum()
    print(f"total de NaNs após o processamento: {nans}")

    # Salvando o df limpo em .csv novamente
    try:
        output_dir = os.path.dirname(output_path) # Garante que o diretório de saída exista
        os.makedirs(output_dir, exist_ok=True)
        clean_df.to_csv(output_path, index=False)
        print(f"\nDataset salvo com sucesso em: {output_path}")
    except (IOError, PermissionError) as e:
        print(f"\nERRO: Não foi possível salvar o arquivo em '{output_path}'. Erro: {e}")
        raise e

    return clean_df