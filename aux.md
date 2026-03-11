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

`python multi_output_experiments.py --model_name nome_do_modelo --dataset_name nome_do_dataset`

`python regressor_chain_experiments.py --model_name nome_do_modelo --dataset_name nome_do_dataset`

`python native_experiments.py --model_name nome_do_modelo --dataset_name nome_do_dataset`


----------------------------------------------------------------------------------
## A seguir:

- terminar o exploring results:
    executar o script no servidor para obter os dataframes processados.

- Validação "In-Loco" 
Objetivo: Provar que o modelo treinado no Birds 
realmente sabe ranquear os pipelines do Birds


    Separar o dataset de teste original (dados reais)
    Filtrar apenas os 10% melhores pipelines (F1 mais alto)
    XGBoost ordenar esses pipelines
    Calcular a Correlação de Spearman entre a ordem real e a ordem prevista
    
Prova de Conceito (PoC):
    Usar o script de geração aleatória 
    Selecionar o Top 3 Pipelines sugeridos pelo modelo
    Implementar esse pipeline manualmente e rodar no dataset original Birds
    Comparar o F1 Real vs. F1 Previsto. 


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


