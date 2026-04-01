import pygad
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

class GAPipelineOptimizer:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        
        # Carregar recursos do seu projeto
        self.model = joblib.load(self.BASE_DIR / "candidate_models" / dataset_name / "xgboost_multi_output.joblib")
        
        with open(self.BASE_DIR / "configs" / "feature_schema.json", 'r') as f:
            self.columns = json.load(f)
        with open(self.BASE_DIR / "configs" / "global_hyperparameter_ranges.json", 'r') as f:
            self.ranges = json.load(f)
            
        self.col_to_idx = {col: i for i, col in enumerate(self.columns)}
        
        # Mapear grupos (Mesma lógica do seu PipelineGenerator)
        self.groups = {
            'meka': [c for c in self.columns if c.startswith('meka.') and '-' not in c],
            'mlfs_sk': [c for c in self.columns if (c.startswith('mlfs.') or c.startswith('sklearn.')) and '-' not in c],
            'weka': [c for c in self.columns if c.startswith('weka.') and '-' not in c]
        }

    def repair_and_discretize(self, dna):
        """ Garante que o DNA respeite as leis do seu projeto """
        new_dna = dna.copy()
        
        # 1. Regra de Ouro: Preprocessing sempre ATIVO (1.0)
        new_dna[self.col_to_idx['feature preprocessing']] = 1.0
        
        # 2. Restrição de Grupos (Apenas 1 algoritmo por grupo principal)
        # Para cada grupo, mantemos apenas o gene com maior valor e zeramos o resto
        for group_name, cols in self.groups.items():
            indices = [self.col_to_idx[c] for c in cols]
            max_idx = indices[np.argmax(new_dna[indices])]
            for idx in indices:
                new_dna[idx] = 1.0 if idx == max_idx else 0.0
                
        # 3. Discretização e Clamping (Ranges)
        for i, col_name in enumerate(self.columns):
            # Se for um parâmetro ativo (tem '-' no nome)
            if '-' in col_name:
                parent_alg = col_name.split('-')[0]
                # Só processamos se o algoritmo pai estiver ativo (1.0)
                if new_dna[self.col_to_idx[parent_alg]] == 1.0:
                    if col_name in self.ranges:
                        p_min = self.ranges[col_name]['min']
                        p_max = self.ranges[col_name]['max']
                        # Garante que o AG não saia do limite do JSON
                        val = np.clip(new_dna[i], p_min, p_max)
                        
                        # --- DENTRO DO SEU LOOP DE REPARO ---

                        # 1. Primeiro tratamos a exceção das Features (Mínimo de 10)
                        if 'n_features' in col_name:
                            new_dna[i] = float(max(10, int(round(val))))

                        # 2. Depois tratamos os outros que precisam ser INTEIROS (sem o n_features aqui)
                        elif any(x in col_name for x in ['Neighbors', '-I', '-K', '-depth', '-B']):
                            new_dna[i] = float(max(1, int(round(val)))) # Garante ao menos 1 para vizinhos/árvores

                        # 3. Para o restante (parâmetros contínuos como Learning Rate, E, M, etc)
                        else:
                            new_dna[i] = val
                    else:
                        new_dna[i] = -1.0 # Desativa parâmetro de algoritmo não escolhido
                    
        return new_dna

    def fitness_func(self, ga_instance, solution, solution_idx):
        # Repara o DNA antes de avaliar
        clean_dna = self.repair_and_discretize(solution)
        
        # Predição (XGBoost)
        pred = self.model.predict(clean_dna.reshape(1, -1))
        f1_score = pred[0][0]
        
        # Penalidade leve por pipelines muito grandes (Opcional)
        # f1_score -= (clean_dna.sum() * 0.0001) 
        
        return float(f1_score)

# --- EXECUÇÃO ---
# --- EXECUÇÃO ---
optimizer = GAPipelineOptimizer("birds")

# 1. Espaço Genético
gene_space = []
for col in optimizer.columns:
    if '-' in col:
        p_min = optimizer.ranges.get(col, {}).get('min', -1.0)
        p_max = optimizer.ranges.get(col, {}).get('max', 1.0)
        gene_space.append({'low': p_min, 'high': p_max})
    else:
        gene_space.append([0.0, 1.0])

# 2. População Inicial (Elite + Aleatórios para completar 200)
elite_path = optimizer.BASE_DIR / "experiments_results" / "predicted_ranking" / "best_candidates_birds.csv"
elite_df = pd.read_csv(elite_path)
elite_data = elite_df.drop(columns=['predicted_F1', 'predicted_model_size_log'], errors='ignore').values

num_needed = 200 - len(elite_data)
random_part = np.random.uniform(low=-1.0, high=1.0, size=(num_needed, len(optimizer.columns)))
combined_population = np.vstack((elite_data, random_part))

# 3. GA Instance
ga_instance = pygad.GA(
    num_generations=200,
    num_parents_mating=20,
    fitness_func=optimizer.fitness_func,
    sol_per_pop=200,               # Agora bate com o tamanho de combined_population
    initial_population=combined_population, 
    gene_space=gene_space,
    parent_selection_type="tournament",
    crossover_type="uniform",
    mutation_probability=0.15,
    on_generation=lambda ga: print(f"🧬 Geração {ga.generations_completed}: Melhor F1 = {ga.best_solution()[1]:.4f}"),
    stop_criteria="saturate_20"    # Aumentei para 20 para dar mais chance de evolução
)

ga_instance.run()

# 4. Resultado
solution, fitness, idx = ga_instance.best_solution()
vencedor_real = optimizer.repair_and_discretize(solution)

print(f"\n🏆 F1 FINAL DA EVOLUÇÃO: {fitness:.4f}")

# 3. Mostrar apenas o que está ativo e conferir os valores
for i, col in enumerate(optimizer.columns):
    val = vencedor_real[i]
    if val != 0.0 and val != -1.0:
        # Se for um parâmetro, vamos checar se ele está "inteiro" como deveria
        if '-' in col:
            print(f"⚙️ Parâmetro: {col} = {val}")
        else:
            print(f"✅ Algoritmo: {col}")