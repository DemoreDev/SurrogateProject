# COM TIMEOUT PARA FEATURE PREPROCESSING - THREADPOLLEXECUTOR


import os

import time

import arff

import tempfile

import numpy as np
import pandas as pd

from sklearn import set_config
#set_config(transform_output="pandas")

import subprocess # MEKA
from multiprocessing import Process, Manager # FP
from multiprocessing.pool import ThreadPool # Pool of run

import mlfs
import sklearn
from mlfs.NO_FEATURE_PREPROCESSING import NO_FEATURE_PREPROCESSING

# meka_adapted4: com timeout e tamanho do classificador
from meka.meka_adapted4 import MekaAdapted

from pymoo.core.problem import Problem
from multiobjective.IndividualUtils import IndividualUtils

from utils.HistoryRun import HistoryRun

import traceback
import mlfs



class MLProblem(Problem):

    def __init__(self, n_gene, k_folds, n_threads, fp_limit_time, mlc_limit_time, config, java_command, meka_classpath, n_labels, n_features, is_dataset_sparse, name_dataset, path_dataset, path_history):
        super().__init__(n_var=n_gene, n_obj=2) 

        self.config = config
        self.java_command = java_command
        self.meka_classpath = meka_classpath
        self.n_labels = n_labels
        self.n_features = n_features
        self.is_dataset_sparse = is_dataset_sparse
        self.name_dataset = name_dataset
        self.path_dataset = path_dataset
        self.path_history = path_history

        self.k_folds = k_folds
        self.n_threads = n_threads
        self.mlc_limit_time = mlc_limit_time  
        self.fp_limit_time = fp_limit_time
        self.temp_datasets = path_dataset+'/tempdatasets'

        self.map_cmd_objectives = {} # cmd: (f1 médio, model size médio)

        # counts
        self.n_queries = 0
        self.n_fp_timeouts = 0
        self.n_mlc_timeouts = 0
        self.n_mlc_exceptions = 0
        self.n_fp_exceptions = 0


    # ARFF -------------------------------------------------------------------------


    def __read_dataset(self, path_dataset):
        arff_frame = arff.load(
            open(path_dataset, 'r'), encode_nominal=False, return_type=arff.DENSE  
        )

        attributes_names_types = np.array(arff_frame['attributes'], dtype=object)
        attributes_names = [attr_name for attr_name, attr_type in attributes_names_types]
        values = np.array(arff_frame['data'], dtype=object)

        X = values[:, :-self.n_labels]
        y = values[:, -self.n_labels:]

        dfX = pd.DataFrame(X, columns=attributes_names[:-self.n_labels])
        dfy = pd.DataFrame(y, columns=attributes_names[-self.n_labels:])

        return attributes_names_types, dfX, dfy 
    

    def __save_new_arrf(self, dfX, dfy, feature_names_types, pfx):
        relation = self.name_dataset+': -C -'+str(self.n_labels)

        attributes = []
        #for i in range(len(dfX.dtypes)):
        #    attributes.append((dfX.dtypes.index[i], 'NUMERIC')) 
        # X
        for attr in dfX.columns:
            # busca tipo
            flag = False
            for name_attr, type_attr in feature_names_types:
                if attr == name_attr:
                    attributes.append((name_attr, type_attr)) 
                    flag = True
                    break

            # caso não encontra tipo, seta atributo como numérico (PCA ...)
            if flag == False:
                attributes.append((attr, 'NUMERIC')) 
        # y
        for i in range(len(dfy.dtypes)): 
            attributes.append((dfy.dtypes.index[i], ['0', '1']))

        df_concat = pd.concat([dfX, dfy], axis=1)
        values = df_concat.to_numpy(dtype=object)

        arff_frame = {}
        arff_frame['relation'] = relation
        arff_frame['attributes'] = attributes
        arff_frame['data'] = values

        arff_save = arff.dumps(arff_frame)
        # pprint.pprint(arff_save)

        temp_file = tempfile.NamedTemporaryFile(dir=self.temp_datasets, prefix=pfx, suffix='.arff', delete=False)
        with open(temp_file.name, 'w', encoding='utf8') as fp:
            fp.write(arff_save)

        return temp_file.name
    

    def __removeArffTempFiles(self, path_tempfile):
        os.unlink(path_tempfile)


    # ARFF -------------------------------------------------------------------------

    # Feature Preprocessinf = process ----------------------------------------------

    def feature_preprocessing_process(fp_algorithm, X, y, result_dict):
        try:
            # Constroi objeto de feature preprocessing
            fp = eval(fp_algorithm) 
            print(fp)

            # Realizando a seleção de características
            fp = fp.fit(X, y)
            
            # Simulando um tempo de processamento mais longo
            # time.sleep(120) 
            
            # Armazenando o resultado no dicionário compartilhado
            result_dict['result'] = fp
        except Exception as e:
            print(traceback.format_exc())
            result_dict['error'] = e


    # Função para executar a seleção de características com timeout
    def run_with_timeout(fp_algorithm, X, y, timeout=5):
        
        with Manager() as manager:
            # Dicionário compartilhado entre os processos
            result_dict = manager.dict()
            
            # Criando o processo
            process = Process(target=MLProblem.feature_preprocessing_process, args=(fp_algorithm, X, y, result_dict))
            
            # Iniciando o processo
            process.start()
            
            # Esperando o processo finalizar ou o timeout
            process.join(timeout)
            
            # Se o processo ainda está vivo após o timeout
            if process.is_alive():
                print("FP Timeout! O processo foi interrompido.")
                process.terminate()
                process.join()  
                return None # timeout
            else:
                if 'error' in result_dict:
                    print(f"Erro no algoritmo de seleção de características: {result_dict['error']}")
                    return result_dict['error'] # exception
                
                # Feature preprocessing sem erro e sem timeout
                return result_dict.get('result', None)

    # Feature Preprocessinf = process ----------------------------------------------
    #   
    
    def my_eval(self, param):
        print('Iniciando ...')

        # configuração no contexto da thread
        set_config(transform_output="pandas")
        
        is_normalize, fp_command, meka_command, weka_command = param
        #print('Normalize: {}'.format(is_normalize))
        #print('Feature Prep.: {}'.format(fp_command))
        #print('MEKA: {}'.format(meka_command))
        #print('WEKA: {}'.format(weka_command))

        command = str(is_normalize) + ' ' + fp_command + ' ' + meka_command
        if weka_command is not None:
            command = command + ' ' + weka_command
            
        # consulta mapa de objetivos para o comando 
        # se comando já foi avaliado nos k folds  
        # utiliza os valores médios dos objetivos já calculados
        if command in self.map_cmd_objectives.keys():   
            f1_mean = self.map_cmd_objectives.get(command)[0] 
            ms_mean = self.map_cmd_objectives.get(command)[1]
            self.n_queries += 1  
            print(f'Algorithm already evaluated: {f1_mean} {ms_mean}')

        else:
            # prepara comando meka
            meka = MekaAdapted(
                meka_classifier = meka_command,
                weka_classifier = weka_command,
                meka_classpath = self.meka_classpath, 
                java_command = self.java_command,
                timeout = self.mlc_limit_time
            )

            # obtém nome dos datasets
            if is_normalize == False:
                train_dataset = self.path_dataset+'/'+self.name_dataset+'-train-'
                test_dataset = self.path_dataset+'/'+self.name_dataset+'-test-'
            else:
                train_dataset = self.path_dataset+'/'+self.name_dataset+'-norm-train-'
                test_dataset = self.path_dataset+'/'+self.name_dataset+'-norm-test-'

            # armazena valores dos objetivos para os k-folds
            list_f1 = np.array([], dtype=float)
            list_ms = np.array([], dtype=float)

            # executa algoritmo para os k-folds
            for k in range(self.k_folds):
                # tempfiles
                train_file = ''
                test_file = ''
                model_size = 0
                features_selected = 0
                statistics = {}

                try:
                    # atualiza nomes dos datasets com 'k'
                    train_dataset_k = train_dataset+str(k)+'.arff' 
                    test_dataset_k = test_dataset+str(k)+'.arff' 

                    # lê datasets
                    feature_names_types, X_train, y_train = self.__read_dataset(train_dataset_k)
                    _, X_test, y_test = self.__read_dataset(test_dataset_k)
                    #print(f'1) X (tr/te): {X_train.shape}, {X_test.shape}')

                    # obtém algoritmo de feature selection
                    fp = MLProblem.run_with_timeout(fp_command, X_train, y_train, self.fp_limit_time)

                    # feature preprocessing result
                    if fp:   
                        if isinstance(fp, Exception):
                            self.n_fp_exceptions += 1
                        else:          
                            # aplica algoritmo de FP nos dados de entrada
                            X_train = fp.transform(X_train)
                            X_test = fp.transform(X_test)
                            #print(f'2) X (tr/te): {X_train.shape}, {X_test.shape}')

                            # salva arff temporário após FP
                            train_file = self.__save_new_arrf(X_train, y_train, feature_names_types, 'train-'+str(k)+'-')
                            test_file = self.__save_new_arrf(X_test, y_test, feature_names_types, 'test-'+str(k)+'-')

                            # treina o modelo MLC
                            n_test_example = X_test.shape[0]
                            meka.fit_predict(n_test_example, self.config.n_labels, train_file, test_file)   # TimeoutExpired

                            # medidas
                            model_size = meka.len_model_file 
                            features_selected = X_train.shape[1]
                            statistics = meka.statistics
                            f1 = statistics.get('F1 (macro averaged by label)')

                            # adiciona resultados do fold na lista
                            list_f1 = np.append(list_f1, f1)
                            list_ms = np.append(list_ms, model_size)
                    else:
                        # feature preprocessing timeout
                        self.n_fp_timeouts += 1

                except subprocess.TimeoutExpired as e:
                    print('MEKA Timeout! O processo foi interrompido.') 
                    self.n_mlc_timeouts += 1

                except Exception as e:
                    print('---------- Exception ----------\n')
                    print('Normalize: {}'.format(is_normalize))
                    print('Feature Prep.: {}'.format(fp_command))
                    print('MEKA: {}'.format(meka_command))
                    print('WEKA: {}\n'.format(weka_command))
                    #print(e)
                    print(traceback.format_exc())
                    print('---------- Exception ----------\n')
                    self.n_mlc_exceptions += 1
                
                finally:
                    # remove arff temporário
                    if train_file != '':
                        self.__removeArffTempFiles(train_file)
                    if test_file != '':
                        self.__removeArffTempFiles(test_file)
                        
                HistoryRun.add_metrics(k, is_normalize, fp_command, meka_command, weka_command, statistics, model_size, features_selected)

            if len(list_f1) > 0:
                f1_mean = list_f1.mean()
                ms_mean = list_ms.mean()
            else:
                f1_mean = 0
                ms_mean = 1e9

            print(list_f1, f1_mean)
            print(list_ms, ms_mean)
            self.map_cmd_objectives[command] = (f1_mean, ms_mean) 

        return [-f1_mean, ms_mean]


    def _evaluate(self, X, out, *args, **kwargs): 
        pool = ThreadPool(self.n_threads)
        
        # prepare the parameters for the pool
        params = [[IndividualUtils.get_commands(self.config, X[n])] for n in range(len(X))]
        
        # pool de threads
        F = pool.starmap(self.my_eval, params)
        
        pool.close()  

        # store the function values and return them.
        out["F"] = np.array(F, dtype=object)

        HistoryRun.to_file(self.name_dataset, self.path_history)