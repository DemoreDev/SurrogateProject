import numpy as np

import configuration.SLC as config_SLC
import configuration.SLCEnsemble as config_SLC_ensemble

    
def get_config(n_features, n_labels, weighted_instances_handler = False, only_multiclass_classifiers = False):
    payoff_function = np.array(['Accuracy', '\'Jaccard index\'', '\'Hamming score\'', '\'Exact match\'',
                    '\'Jaccard distance\'', '\'Hamming loss\'', '\'ZeroOne loss\'',
                    '\'Harmonic score\'', '\'One error\'', '\'Rank loss\'', '\'Avg precision\'',
                    '\'Log Loss (lim. L)\'', '\'Log Loss (lim. D)\'', '\'Micro Precision\'',
                    '\'Micro Recall\'', '\'Macro Precision\'', '\'Macro Recall\'',
                    '\'F1 (micro averaged)\'', '\'F1 (macro averaged by example)\'',
                    '\'F1 (macro averaged by label)\'', '\'AUPRC (macro averaged)\'',
                    '\'AUROC (macro averaged)\'', '\'Levenshtein distance\''])
    
    MLC_config_dic = {


        #java -cp meu-meka-1.9.8/lib/* meka.classifiers.multilabel.MULAN -S MLkNN.15 -t emotions-train-0.arff -T emotions-test-0.arff
        #java -cp meu-meka-1.9.8/lib/* meka.classifiers.multilabel.MULAN -S HOMER.Random.3.BinaryRelevance -t emotions-train-0.arff -T emotions-test-0.arff -verbosity 5


        'meka.classifiers.multilabel.MULAN.MLkNN': {
            '-normalize': np.array([False, True], dtype=bool),
            '-numOfNeighbors': np.arange(1, 65, dtype=int)
        },

        'meka.classifiers.multilabel.MULAN.HOMER': {
            '-method': np.array(['BalancedClustering', 'Clustering', 'Random']),
            '-clusters': np.arange(2, n_labels+1 if n_labels < 9 else 8, dtype=int),
            '-mll': np.array(['BinaryRelevance', 'ClassifierChain', 'LabelPowerset'])
        },

        # default hyperparameters
        'meka.classifiers.multilabel.MULAN.ECC': {
        },

        # --------------------

        'meka.classifiers.multilabel.BPNN': {
            '-normalize': np.array([False, True], dtype=bool),
            '-E': np.arange(10, 1001, dtype=int),
            '-H': np.arange(int(0.2 * n_features), n_features + 1, dtype=int),
            '-r': np.around(np.arange(0.001, 0.1001, 0.001), 3),
            '-m': np.around(np.arange(0.1, 0.9, 0.1), 1),
        },
        

        'meka.classifiers.multilabel.BR': {
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers)  
        },


        'meka.classifiers.multilabel.CC': {
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers)  
        },


        'meka.classifiers.multilabel.RAkEL': {
            '-N': np.arange(0, 6, dtype=int),
            '-P': np.arange(1, 6, dtype=int),
            '-k': np.arange(1, int(n_labels/2) + 1, dtype=int),
            '-M': np.arange(2, min(2 * n_labels, 100) + 1, dtype=int),
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True)
        },
        
        
        'meka.classifiers.multilabel.LC': { # LP
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True)    
        },


        'meka.classifiers.multilabel.BCC': {
            '-X': np.array(['C', 'I', 'Ib', 'Ibf', 'H', 'Hbf', 'X', 'F', 'L', 'None']), 
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers) 
        },


        'meka.classifiers.multilabel.BRq': {
            '-P': np.around(np.arange(0.1, 0.805, 0.05), 2),
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers) 
        },
        

        'meka.classifiers.multilabel.FW': {
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True) 
        },

        'meka.classifiers.multilabel.MCC': {
            '-Iy': np.arange(1, 101, 1, dtype=int),
            '-Is': np.append(np.zeros(1499, dtype=int), np.arange(2, 1501, 1, dtype=int)),
            '-P': payoff_function,            
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers) 
        },
        

        # viável para problemas com menos de 15 rótulos
        'meka.classifiers.multilabel.PCC': {     
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers) 
        },


        'meka.classifiers.multilabel.PS': {    
            '-P': np.arange(1, 6, dtype=int),
            '-N': np.arange(0, 6, dtype=int),
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True) 
        },
        

        'meka.classifiers.multilabel.PSt': {    
            '-P': np.arange(1, 6, dtype=int),
            '-N': np.arange(0, 6, dtype=int),
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True) 
        },


        'meka.classifiers.multilabel.RAkELd': {    
            '-P': np.arange(1, 6, dtype=int),
            '-N': np.arange(0, 6, dtype=int),
            # -M não é hiperparâmetro de RAkELd
            '-batch-size': np.arange(2, min(2 * n_labels, 100) + 1, dtype=int), 
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True) 
        },


        'meka.classifiers.multilabel.RT': {     
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True) 
        },


        'meka.classifiers.multilabel.CT': {
            '-X': np.array(['C', 'I', 'Ib', 'Ibf', 'H', 'Hbf', 'X', 'F', 'L', 'None']), 
            '-Iy': np.arange(1, 101, 1, dtype=int),
            '-Is': np.append(np.zeros(1499, dtype=int), np.arange(2, 1501, 1, dtype=int)),
            '-P': payoff_function,
            '-H': np.array([-1, 0, 1]),
            'if': lambda params: {'-L': np.arange(1, np.sqrt(n_labels) + 2, dtype=int)} if params['-H']==-1 else None,
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers)
        },


        'meka.classifiers.multilabel.CDN': {
            '-Ic': np.arange(1, 101, 1, dtype=int),
            '-I': np.arange(100, 1001, 1, dtype=int),
            # I > IC
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers)
        },
        

        'meka.classifiers.multilabel.CDT': {
           '-H': np.array([-1, 0, 1]),
            '-X': np.array(['C', 'I', 'Ib', 'Ibf', 'H', 'Hbf', 'X', 'F', 'L', 'None']), 
            '-I': np.arange(1, 1001, 1, dtype=int),
            '-Ic': np.arange(1, 101, 1, dtype=int),
            # I > IC
            'if': lambda params: {'-L': np.arange(1, np.sqrt(n_labels) + 2, dtype=int)} if params['-H']==-1 else None,
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers)
        },
        
         
        'meka.classifiers.multilabel.PMCC': {
            '-Iy': np.arange(1, 101, 1, dtype=int),
            '-Is': np.arange(50, 1501, 1, dtype=int), # chain iterations
            '-B': np.arange(0.01, 1, 0.01),
            '-O': np.array([0, 1], dtype=int),
            '-M': np.arange(1, 51, dtype=int), # population size
            # -M < -Is
            '-P': payoff_function,            
            '-W': get_slc_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers) 
        }

    }
    
    return MLC_config_dic

def get_MLC_algorithms(n_features, n_labels, weighted_instances_handler=False):
    return np.array(list(get_config(n_features, n_labels, weighted_instances_handler).keys()))


def get_slc_config(n_features, n_labels, weighted_instances_handler=False, only_multiclass_classifiers=False):
    return {**config_SLC_ensemble.get_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers), 
            **config_SLC.get_config(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers)}  
