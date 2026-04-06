from sklearn.base import BaseEstimator, TransformerMixin

from mlfs.base_algorithms.lrfs import lrfs

from sklearn.preprocessing import KBinsDiscretizer

import numpy as np


class PyIT_LRFS(BaseEstimator, TransformerMixin):

    def __init__(self, n_features):
        self.n_features = int(n_features)
        self.index_ranks = None
        
        
    def __str__(self):
        return 'PYIT_LRFS('+str(self.n_features)+')'
    
    
    def fit(self, dfX, dfy=None):   
        ## discretizer
        kb_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        kb_discretizer.fit(dfX)
        # a saída é um dataframe por conta da configuração: set_config(transform_output="pandas") 
        new_dfX = kb_discretizer.transform(dfX)

        X = np.asarray(new_dfX, dtype=int)
        y = np.asarray(dfy, dtype=int)

        self.index_ranks = lrfs().rank(X, y)
        return self
    

    def transform(self, dfX, dfy = None):
        indexes = self.index_ranks[:self.n_features]
        return dfX.iloc[:, indexes]