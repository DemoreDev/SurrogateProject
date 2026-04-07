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

### **Para treinar um modelo de uma abordagem específica**
```
python multi_output_experiments.py --model_name nome_do_modelo --dataset_name nome_do_dataset

python regressor_chain_experiments.py --model_name nome_do_modelo --dataset_name nome_do_dataset

python native_experiments.py --model_name nome_do_modelo --dataset_name nome_do_dataset
```

### **Para gerar pipelines candidatos**
```
for i in {1..5}; do
python3 candidate_generation_experiment.py --model_name xgboost --dataset_name medical --n_instances 50000 --batch_id $i
done
```

### **Para validar pipelines preditos**
```
python3 validation_experiment.py --dataset_name medical --top_n 1
```

----------------------------------------------------------------------------------
## A seguir:

Começar o script de validação de candidatos

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


Análise Comparativa e Explicabilidade
Objetivo: Adicionar profundidade e entender
por que o modelo faz suas escolhas

Explicabilidade com SHAP:
    Rodar o SHAP (shap.TreeExplainer) no XGBoost
    Gerar o Summary Plot: Mostre quais hiperparâmetros são mais vitais
    Crucial: achar interações entre Meta-Features e Hiperparâmetros
    Exemplo: O gráfico de dependência mostra que quando nr_instances é baixo, 
    o modelo penaliza rf_n_estimators alto (evitando overfitting)? 



