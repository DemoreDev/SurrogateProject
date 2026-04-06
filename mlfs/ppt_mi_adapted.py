from sklearn.base import BaseEstimator, TransformerMixin

from mlfs.base_algorithms.ppt_mi import ppt_mi

from sklearn.preprocessing import KBinsDiscretizer

import numpy as np


class PyIT_PPT_MI(BaseEstimator, TransformerMixin):

    def __init__(self, n_features, prune_threshold=6):
        self.n_features = int(n_features)
        self.prune_threshold = prune_threshold
        self.index_ranks = None
        
        
    def __str__(self):
        return 'PYIT_PPT_MI('+str(self.n_features)+', '+str(self.prune_threshold)+')'
    
    
    def fit(self, dfX, dfy=None):  
        # discretizer
        kb_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        kb_discretizer.fit(dfX)
        # a saída é um dataframe por conta da configuração: set_config(transform_output="pandas") 
        new_dfX = kb_discretizer.transform(dfX)

        X = np.asarray(new_dfX, dtype=int)
        y = np.asarray(dfy, dtype=int) 

        self.index_ranks = ppt_mi().rank(X, y)
        return self
    

    def transform(self, dfX, dfy = None):
        indexes = self.index_ranks[:self.n_features]
        return dfX.iloc[:, indexes]