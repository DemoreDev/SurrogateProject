import os
import json
from pathlib import Path

# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Define os caminhos do projeto
JSON_PATH = BASE_DIR / "json_configs" / "global_hyperparameter_ranges.json"

# Caminho fixo do Java Classpath que já validamos
MEKA_CLASSPATH = str(BASE_DIR / "lib" / "*")

def load_hyperparameters_json() -> dict:
    # Carrega o JSON de hyperparams (e tipos) para um dicionário

    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Warning: JSON file not found at {JSON_PATH}.")
        return {}