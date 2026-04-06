## Abordagens e modelos escolhidos para treinar:
- MultiOutputRegressor:
    - Ridge 
    - RandomForestRegressor 
    - LGBMRegressor 
    - CatBoostRegressor 
    - XGBoostRegressor 

- RegressorChain
    - RandomForestRegressor 
    - LGBMRegressor 
    - XGBoostRegressor

- Adaptação Nativa e Abordagens Alternativas:
    - RandomForestRegressor 
    - KNeighborsRegressor 
    - CatBoost
    - MLPRegressor


## Como rodar os experimentos no terminal:

```
python multi_output_experiments.py --model_name nome_do_modelo --dataset_name nome_do_dataset

python regressor_chain_experiments.py --model_name nome_do_modelo --dataset_name nome_do_dataset

python native_experiments.py --model_name nome_do_modelo --dataset_name nome_do_dataset
```

```
for i in {1..20}; do
python3 candidate_generation_experiment.py --model_name xgboost --dataset_name medical --n_instances 50000 --batch_id $i
done
```


----------------------------------------------------------------------------------
## A seguir:

- A Generalização 
Objetivo: Criar um modelo capaz de operar em datasets 
que ele nunca viu usando Meta-Features

Extração de Meta-Features:
    Instalar biblioteca pymfe 
    Para cada um dos 5 datasets originais, extrair um vetor de características 
        Simples: Número de instâncias, número de atributos, proporção de classes.
        Estatísticas: Média da correlação, Skewness (assimetria), Kurtosis.
        Gerar um df: dataset_id | meta_feature_1 | meta_feature_2 | ...

Criação do Dataset-mestre:
        CSVs de treino atuais (meta-datasets)
        Adicionar as colunas das meta-features em todas as linhas
        (todas as linhas do CSV Medical ganham os valores das meta-features do Medical)
        Concatenar 4 CSVs em um arquivo gigante (deixar 1 de fora para teste)
        Treinar o XGBoost nos outros 4 (dataset-mestre)
        Testar a performance no que ficou de fora
        Repetir trocando o dataset de teste (generalização)

- O Motor de Otimização ϵ-Greedy

Objetivo: Implementar o algoritmo que usa o surrogado para 
encontrar a melhor solução, substituindo a busca cega

Implementação da Busca:
    Criar função que gera vários pipelines candidatos aleatórios (ordem de 10000)
    Passar os candidatos pelo modelo generalista (já alimentado com as meta-features do dataset alvo)

Lógica ϵ-Greedy (Epsilon-Greedy):
    Definir um ϵ (ex: 0.10)
        Gerar um número aleatório:
        Se < 0.90: Escolher o candidato com o maior F1 previsto 
        Se < 0.10: Escolher um candidato totalmente aleatório 

Benchmark de Eficiência:
    Comparar ϵ-Greedy vs. Random Search 


Análise Comparativa e Explicabilidade
Objetivo: Adicionar profundidade e entender
por que o modelo faz suas escolhas

Explicabilidade com SHAP:
    Rodar o SHAP (shap.TreeExplainer) no XGBoost
    Gerar o Summary Plot: Mostre quais hiperparâmetros são mais vitais
    Crucial: achar interações entre Meta-Features e Hiperparâmetros
    Exemplo: O gráfico de dependência mostra que quando nr_instances é baixo, 
    o modelo penaliza rf_n_estimators alto (evitando overfitting)? 

Empacotamento Final

Tabelas Finais:
    Ranking dos Modelos (já tenho)
    Tabela de Generalização (Performance Leave-One-Out)
    Comparativo de Tempo: ϵ-Greedy vs Random Search 

Pontos: 
    - análise de modelos (Fase 1)
    - modelo escalável (Fase 2)
    - método de busca eficiente (Fase 3) 
    - explicação (Fase 4)


