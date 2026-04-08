
import numpy as np

def get_config(n_features, n_labels, weighted_instances_handler=False, only_multiclass_classifiers=False, randomizable=False):
    
    SLC_config_dic = {

        'weka.classifiers.bayes.NaiveBayes': {
            '-normalize': np.array([False], dtype=bool),
            '-K': np.array([False, True], dtype=bool),
            'if': lambda params: {'-D': np.array([False, True], dtype=bool)} if params['-K']==False else None
        }, 


        'weka.classifiers.rules.PART': {
            '-normalize': np.array([False], dtype=bool),
            '-M': np.arange(1, 65, dtype=int),
            '-B': np.array([False, True], dtype=bool),
            '-R': np.array([False, True], dtype=bool),
            'if': lambda params: {'-N': np.array([2,3,4,5])} if params['-R']==True else None
        }, 
        
         
        # C4.5
        'weka.classifiers.trees.J48': {
            '-normalize': np.array([False], dtype=bool),
            '-M': np.arange(1, 65, dtype=int),
            '-B': np.array([False, True], dtype=bool),
            '-J': np.array([False, True], dtype=bool),
            '-A': np.array([False, True], dtype=bool),
            '-U': np.array([False, True], dtype=bool), # T unpruned
            'if': lambda params: {'-C': np.arange(0.05, 1, 0.05), '-O': np.array([False, True], dtype=bool), '-S': np.array([False, True], dtype=bool)} if params['-U']==False else None 
            # -C confidence factor
            # -O collapse tree
            # -S subtree raising
        },
         
         
        'weka.classifiers.lazy.IBk': {
            '-normalize': np.array([False, True], dtype=bool),
            '-K': np.arange(1, 65, dtype=int),
            '-X': np.array([False, True]),
            '-I': np.array([False, True]),
            'if': lambda params: {'-F': np.array([False, True])} if params['-I']==False else None
        },
         
         
        'weka.classifiers.functions.SMO': {
            '-normalize': np.array([False, True], dtype=bool),
            '-C': np.arange(0.5, 1.6, 0.1),
            '-N': np.array([0,1,2]),
            '-M': np.array([False, True], dtype=bool),
            # kernel
            '-K': get_kernels_smo()
        },
         
         
        'weka.classifiers.rules.JRip': {
            '-normalize': np.array([False], dtype=bool),
            '-N': np.around(np.arange(1, 5.1, 0.1), 1),
            # D
            '-E': np.array([False, True], dtype=bool),
            '-P': np.array([False, True], dtype=bool),
            '-O': np.arange(1, 6, dtype=int)
        },                
         

        'weka.classifiers.functions.Logistic': {
            '-normalize': np.array([False], dtype=bool),
            '-R': np.array([10**(-x) for x in range(12, -2,-1)])
        },
         
         
        'weka.classifiers.bayes.BayesNet': {
            '-normalize': np.array([False], dtype=bool),
            '-D': np.array([True]), # Do not use ADTree data structure
            '-Q': np.array(['weka.classifiers.bayes.net.search.local.TAN', 
                            'weka.classifiers.bayes.net.search.local.K2 -- -P 1', # INT_P = 1 (Maximum number of parents)
                            'weka.classifiers.bayes.net.search.local.HillClimber -- -P 1', # INT_P = 1
                            'weka.classifiers.bayes.net.search.local.LAGDHillClimber -- -P 1', # INT_P = 1
                            # 'weka.classifiers.bayes.net.search.local.SimulatedAnnealing -- -U 10000', # INT_U = 10000 (Nuber of runs)
                            'weka.classifiers.bayes.net.search.local.TabuSearch -- -P 1']) # INT_P = 1
        },
        

        'weka.classifiers.trees.RandomForest': {
            '-normalize': np.array([False], dtype=bool),
            '-I': np.arange(2, 257, dtype=int),
            '-K': np.append(np.zeros(31, dtype=int), np.arange(2, 33, dtype=int)),
            '-depth': np.append(np.zeros(19, dtype=int), np.arange(2, 21, dtype=int))
        },

         
        'weka.classifiers.rules.DecisionTable': {
            '-normalize': np.array([False], dtype=bool),
            '-E': np.array(['acc', 'rmse', 'mae', 'auc']),
            '-I': np.array([False, True], dtype=bool),
            '-S': np.array(['BestFirst', 'GreedyStepwise']),
            '-X': np.array([1,2,3,4])
        },
         
         
        # weighted
        'weka.classifiers.lazy.KStar': {
            '-normalize': np.array([False, True], dtype=bool),
            '-B': np.arange(1, 101, dtype=int),
            '-E': np.array([False, True], dtype=bool),
            '-M': np.array(['a', 'd', 'm', 'n'])
            # average column entropy curves (a); 
            # ignore the instances with missing values (d)
            # treat missing values asmaximally diﬀerent (m)
            # normalize over the attributes (n).
        },
         
         
        # weighted
        'weka.classifiers.trees.LMT': {
            '-normalize': np.array([False], dtype=bool),
            '-M': np.arange(1, 65, dtype=int),
            '-B': np.array([False, True], dtype=bool),
            '-R': np.array([False, True], dtype=bool),
            '-C': np.array([False, True], dtype=bool),
            '-P': np.array([False, True], dtype=bool),
            '-W': np.arange(0, 1.01, 0.01),
            '-A': np.array([False, True], dtype=bool),
        },
         
         
        'weka.classifiers.functions.MultilayerPerceptron': {
            '-normalize': np.array([False, True], dtype=bool),
            '-L': np.around(np.arange(0.1, 1.1, 0.1), 1),
            '-M': np.around(np.arange(0, 1.1, 0.1), 1),
            '-H': np.array([int(np.around((n_features + n_labels)/2)), n_features, n_labels, n_features + n_labels]),
            '-B': np.array([False, True], dtype=bool),
            '-R': np.array([False, True], dtype=bool),
            '-D': np.array([False, True], dtype=bool)
        },
         
         
        'weka.classifiers.trees.REPTree': {
            '-normalize': np.array([False], dtype=bool),
            '-M': np.arange(1, 65, dtype=int),
            '-L': np.append(-np.ones(19, dtype=int), np.arange(2, 21, dtype=int)),
            '-P': np.array([False, True], dtype=bool)
        },
         
         
        # weighted
        # classificador binário
        'weka.classifiers.functions.SGD': {
            '-normalize': np.array([False], dtype=bool),
            '-F': np.array([0, 1]), # removi o 2
            '-L': np.array([10**(x) for x in range(-5, 0, 1)]),
            '-R': np.array([10**(x) for x in range(-12, 2, 1)]),
            '-N': np.array([False, True], dtype=bool),
            '-M': np.array([False, True], dtype=bool)
        },
         
         
        'weka.classifiers.trees.RandomTree': {
            '-normalize': np.array([False], dtype=bool),
            '-M': np.arange(1, 65, dtype=int),
            '-K': np.append(np.zeros(31, dtype=int), np.arange(2, 33, dtype=int)),
            '-depth': np.append(np.zeros(19, dtype=int), np.arange(2, 21, dtype=int)),
            '-N': np.append(np.zeros(4, dtype=int), np.arange(2, 6, dtype=int)),
        },
         
         
        'weka.classifiers.functions.SimpleLogistic': {
            '-normalize': np.array([False], dtype=bool),
            '-W': np.arange(0, 1.1, 0.1),
            '-S': np.array([False, True], dtype=bool),
            '-A': np.array([False, True], dtype=bool),
        },
         
         
        # classificador binário
        'weka.classifiers.functions.VotedPerceptron': {
            '-normalize': np.array([False, True], dtype=bool),
            '-I': np.arange(1, 10, dtype=int),
            '-M': np.arange(5000, 50001, dtype=int),
            '-E': np.arange(0.2, 5, 0.1)
        },
         
    }
    
    
    if randomizable == True:
        SLC_config_dic = {k: v for k, v in SLC_config_dic.items() if k in [
            'weka.classifiers.trees.RandomForest',
            'weka.classifiers.trees.RandomTree',
            'weka.classifiers.trees.REPTree',
            'weka.classifiers.functions.SGD',
            'weka.classifiers.functions.MultilayerPerceptron']} 
        
        
    if weighted_instances_handler == True:
        SLC_config_dic = {k: v for k, v in SLC_config_dic.items() if k not in [
            'weka.classifiers.trees.LMT',
            'weka.classifiers.lazy.KStar',
            'weka.classifiers.functions.SGD', 
            'weka.classifiers.rules.OneR',
            'weka.classifiers.functions.VotedPerceptron']}
        
        
    if only_multiclass_classifiers == True:  
        SLC_config_dic = {k: v for k, v in SLC_config_dic.items() if k not in [
            'weka.classifiers.functions.SGD', 
            'weka.classifiers.functions.VotedPerceptron']}
        
        
    return SLC_config_dic


def get_kernels_smo():
    config_smo = {
        
        'weka.classifiers.functions.supportVector.PolyKernel': {
            '-E': np.arange(0.2, 5.1, 0.1),
            '-L': np.array([False, True], dtype=bool)
        },
        
        'weka.classifiers.functions.supportVector.NormalizedPolyKernel': {
            '-E': np.arange(0.2, 5.1, 0.1),
            '-L': np.array([False, True], dtype=bool)
        },
        
        'weka.classifiers.functions.supportVector.Puk': {
            '-O': np.arange(0.1, 1.1, 0.1),
            '-S': np.arange(0.1, 10.1, 0.1)
        },
        
        'weka.classifiers.functions.supportVector.RBFKernel': {
            '-G': np.array([10**(x) for x in range(-4, 1, 1)])
        }
        
    }
        
    return config_smo